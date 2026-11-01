"""Fail-closed Layer 4 interlocks: thresholds, data validation, action gating, config."""
import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.interlocks import (
    CLEAR, DATA_INVALID, TRIGGERED, ConfigError, DEFAULT_CONFIG_PATH,
    evaluate_interlocks, load_clinical_profile, load_interlock_config,
)
from src.safety_engine import evaluate_contraindications

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
CFG = load_interlock_config()


def profile(**over):
    p = {
        "vitals": {"HeartRate": 78, "SystolicBP": 118, "collected_at": "2026-11-02T11:50:00Z"},
        "labs": {"eGFR": 88.0, "Creatinine": 0.95, "Platelets": 240000, "INR": 1.05, "Lactate": 1.2,
                 "collected_at": "2026-11-02T06:00:00Z"},
    }
    for k, v in over.items():
        sec = "vitals" if k in ("HeartRate", "SystolicBP") else "labs"
        if v is None:
            p[sec].pop(k, None)
        else:
            p[sec][k] = v
    return p


def run(p, **kw):
    return evaluate_interlocks(p, config=CFG, now=NOW, **kw)


def status(res):
    return {r.rule_id: r.status for r in res.rules}


class Thresholds(unittest.TestCase):
    def test_healthy_all_clear(self):
        res = run(profile())
        self.assertEqual(set(status(res).values()), {CLEAR})
        self.assertEqual(res.issues, ())

    def test_boundaries_do_not_trigger(self):
        # exactly at the threshold is not "<" / ">"; SBP 90 with HR 90 -> SI exactly 1.0
        res = run(profile(eGFR=30, Creatinine=2.0, Platelets=100000, INR=1.7, SystolicBP=90, HeartRate=90, Lactate=2.0))
        self.assertEqual(set(status(res).values()), {CLEAR})

    def test_each_criterion_triggers(self):
        cases = [
            ({"eGFR": 29.9}, "HCE-01"), ({"Creatinine": 2.01}, "HCE-01"),
            ({"Platelets": 99999}, "HCE-02"), ({"INR": 1.71}, "HCE-02"),
            ({"SystolicBP": 89, "HeartRate": 70}, "HCE-03"),   # SBP < 90 rule (absent in old code)
            ({"HeartRate": 121, "SystolicBP": 120}, "HCE-03"),  # SI 1.008
            ({"Lactate": 2.01}, "HCE-03"),
        ]
        for over, rid in cases:
            with self.subTest(over=over):
                self.assertEqual(status(run(profile(**over)))[rid], TRIGGERED)

    def test_old_code_values_no_longer_apply(self):
        # SI 0.95 and lactate 2.1: triggered the old 0.9/2.2 code only for SI; spec says 1.0/2.0
        res = run(profile(HeartRate=114, SystolicBP=120, Lactate=1.9))
        self.assertEqual(status(res)["HCE-03"], CLEAR)

    def test_thresholds_come_from_yaml(self):
        data = copy.deepcopy(CFG.data)
        data["rules"]["HCE-01"]["criteria"][0]["value"] = 45
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text(yaml.safe_dump(data))
            cfg2 = load_interlock_config(p)
        self.assertNotEqual(cfg2.sha256, CFG.sha256)
        res = evaluate_interlocks(profile(eGFR=40), config=cfg2, now=NOW)
        self.assertEqual(status(res)["HCE-01"], TRIGGERED)
        self.assertEqual(res.config_sha256, cfg2.sha256)


class FailClosed(unittest.TestCase):
    def test_no_profile_everything_invalid(self):
        res = run(None)
        self.assertEqual(set(status(res).values()), {DATA_INVALID})
        self.assertEqual(res.issues[0].code, "PROFILE_MISSING")

    def test_missing_field_makes_only_its_rule_invalid(self):
        res = run(profile(Creatinine=None))
        self.assertEqual(status(res), {"HCE-01": DATA_INVALID, "HCE-02": CLEAR, "HCE-03": CLEAR})
        self.assertEqual(res.issues[0].code, "FIELD_MISSING")

    def test_triggered_wins_over_invalid_siblings(self):
        res = run(profile(Lactate=None, SystolicBP=85, HeartRate=70))
        self.assertEqual(status(res)["HCE-03"], TRIGGERED)

    def test_bad_values_are_invalid_not_clear(self):
        for bad, code in [(float("nan"), "FIELD_NOT_FINITE"), (float("inf"), "FIELD_NOT_FINITE"),
                          ("24.5", "FIELD_NOT_NUMERIC"), (True, "FIELD_NOT_NUMERIC"), ([1], "FIELD_NOT_NUMERIC")]:
            with self.subTest(bad=bad):
                res = run(profile(eGFR=bad))
                self.assertEqual(status(res)["HCE-01"], DATA_INVALID)
                self.assertEqual(res.issues[0].code, code)

    def test_unit_conversion_and_unknown_unit(self):
        # 200 umol/L = 2.26 mg/dL > 2.0
        res = run(profile(Creatinine={"value": 200, "unit": "µmol/L"}))
        self.assertEqual(status(res)["HCE-01"], TRIGGERED)
        res = run(profile(Creatinine={"value": 1.0, "unit": "furlongs"}))
        self.assertEqual(res.issues[0].code, "UNIT_UNKNOWN")
        self.assertEqual(status(res)["HCE-01"], DATA_INVALID)

    def test_umol_sent_as_mgdl_is_caught_by_range_check(self):
        res = run(profile(Creatinine=180))
        self.assertEqual(res.issues[0].code, "OUT_OF_RANGE")
        self.assertEqual(status(res)["HCE-01"], DATA_INVALID)

    def test_unitless_assumption_is_recorded(self):
        res = run(profile())
        self.assertTrue(any(w.startswith("UNITLESS_ASSUMED:Creatinine") for w in res.warnings))

    def test_timestamps(self):
        p = profile(); p["labs"]["collected_at"] = "2026-11-01T06:00:00Z"        # 30 h old
        self.assertEqual(run(p).issues[0].code, "STALE")
        p = profile(); del p["labs"]["collected_at"]
        self.assertEqual(run(p).issues[0].code, "TIMESTAMP_MISSING")
        p = profile(); p["labs"]["collected_at"] = "2026-11-02T06:00:00"          # naive
        self.assertEqual(run(p).issues[0].code, "TIMESTAMP_NAIVE")
        p = profile(); p["labs"]["collected_at"] = "yesterday"
        self.assertEqual(run(p).issues[0].code, "TIMESTAMP_INVALID")
        p = profile(); p["vitals"]["collected_at"] = "2026-11-02T13:00:00Z"       # future
        self.assertEqual(run(p).issues[0].code, "FUTURE_TIMESTAMP")
        p = profile(); p["vitals"]["collected_at"] = "2026-11-02T11:00:00Z"       # 60 min > 30
        self.assertEqual(status(run(p))["HCE-03"], DATA_INVALID)

    def test_section_missing(self):
        p = profile(); del p["vitals"]
        res = run(p)
        self.assertEqual(status(res)["HCE-03"], DATA_INVALID)
        self.assertEqual(res.issues[0].code, "SECTION_MISSING")


class ActionGating(unittest.TestCase):
    def test_renal_and_coag_do_not_block_cxr_triage(self):
        res = run(profile(eGFR=20, Creatinine=2.5, Platelets=85000, INR=2.1))
        self.assertEqual([r.rule_id for r in res.blocking_rules(["cxr_triage"])], [])
        self.assertEqual([r.rule_id for r in res.blocking_rules(["contrast_ct"])], ["HCE-01"])
        self.assertEqual([r.rule_id for r in res.blocking_rules(["tpa_recommendation"])], ["HCE-02"])
        self.assertEqual({r.rule_id for r in res.blocking_rules(None)}, {"HCE-01", "HCE-02"})

    def test_hemodynamic_blocks_everything_and_escalates(self):
        res = run(profile(SystolicBP=86, HeartRate=105, Lactate=4.6))
        self.assertEqual([r.rule_id for r in res.blocking_rules(["cxr_triage"])], ["HCE-03"])
        self.assertEqual(res.rules[2].escalation, "EMERGENCY_TRIAGE")

    def test_invalid_data_blocks_like_a_trigger(self):
        res = run(profile(Creatinine=None))
        self.assertEqual([r.rule_id for r in res.blocking_rules(["contrast_ct"])], ["HCE-01"])
        self.assertEqual(res.blocking_rules(["cxr_triage"]), [])

    def test_empty_actions_rejected(self):
        with self.assertRaises(ValueError):
            run(profile()).blocking_rules([])


class ProfileLoading(unittest.TestCase):
    def test_missing_unreadable_and_valid(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prof, issue = load_clinical_profile(d / "nope.json")
            self.assertIsNone(prof); self.assertEqual(issue.code, "PROFILE_MISSING")
            (d / "bad.json").write_text("{not json")
            prof, issue = load_clinical_profile(d / "bad.json")
            self.assertIsNone(prof); self.assertEqual(issue.code, "PROFILE_UNREADABLE")
            (d / "list.json").write_text("[1,2]")
            self.assertEqual(load_clinical_profile(d / "list.json")[1].code, "PROFILE_UNREADABLE")
            (d / "ok.json").write_text(json.dumps(profile()))
            prof, issue = load_clinical_profile(d / "ok.json")
            self.assertIsNone(issue)
            res = run(prof)
            self.assertEqual(set(status(res).values()), {CLEAR})

    def test_load_issue_is_reported(self):
        _, issue = load_clinical_profile("/nonexistent/x.json")
        res = run(None, load_issue=issue)
        self.assertEqual(res.issues[0].code, "PROFILE_MISSING")


class Config(unittest.TestCase):
    def test_shipped_config_matches_spec(self):
        r = CFG.data["rules"]
        crit = lambda rid: {(c["field"], c["op"], c["value"]) for c in r[rid]["criteria"]}
        self.assertEqual(crit("HCE-01"), {("eGFR", "<", 30), ("Creatinine", ">", 2.0)})
        self.assertEqual(crit("HCE-02"), {("Platelets", "<", 100000), ("INR", ">", 1.7)})
        self.assertEqual(crit("HCE-03"), {("SystolicBP", "<", 90), ("ShockIndex", ">", 1.0), ("Lactate", ">", 2.0)})
        self.assertEqual(r["HCE-03"]["blocks"], ["*"])

    def test_invalid_configs_raise(self):
        def broken(mutate):
            d = copy.deepcopy(CFG.data); mutate(d)
            with tempfile.TemporaryDirectory() as t:
                p = Path(t) / "c.yaml"; p.write_text(yaml.safe_dump(d))
                with self.assertRaises(ConfigError):
                    load_interlock_config(p)
        broken(lambda d: d["rules"]["HCE-01"]["criteria"][0].update(op="=="))
        broken(lambda d: d.pop("policy"))
        broken(lambda d: d["rules"]["HCE-02"]["criteria"][0].update(field="Nope"))
        broken(lambda d: d["fields"]["INR"].update(min=10, max=1))
        broken(lambda d: d["rules"]["HCE-03"].update(blocks=[]))
        broken(lambda d: d["rules"]["HCE-01"]["criteria"][0].update(value="30"))

    def test_missing_or_garbage_file_raises(self):
        with self.assertRaises(ConfigError):
            load_interlock_config("/nonexistent/c.yaml")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.yaml"; p.write_text(": : :\n\t- [")
            with self.assertRaises(ConfigError):
                load_interlock_config(p)


class LegacyWrapper(unittest.TestCase):
    def test_wrapper_matches_engine(self):
        p = profile(eGFR=24.5, Creatinine=2.3, Platelets=85000, INR=2.1, SystolicBP=86, HeartRate=105, Lactate=4.6)
        passed, msgs = evaluate_contraindications(p["vitals"], p["labs"], now=NOW)
        self.assertFalse(passed)
        self.assertEqual(len(msgs), 3)
        passed, _ = evaluate_contraindications(p["vitals"], p["labs"], now=NOW, requested_actions=["cxr_triage"])
        self.assertFalse(passed)  # HCE-03 blocks "*"

    def test_missing_dicts_fail_closed(self):
        self.assertFalse(evaluate_contraindications(None, None, now=NOW)[0])


if __name__ == "__main__":
    unittest.main()
