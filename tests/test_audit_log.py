"""Hash-chained audit log + per-scan finalisation (torch-free)."""
import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.audit_log import AuditLog, AuditLogError, GENESIS_HASH, sha256_file
from src.interlocks import load_interlock_config
from src.scan_audit import MANIFEST_FIELDS, append_manifest, finalize_scan, record_pipeline_failure
from tests.test_interlocks import profile

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
CFG = load_interlock_config()
THRESH = {"l1_min_thoracic_pixels": 1000, "l2_aamr_threshold": 0.85}


class Chain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit_log.jsonl"
        self.log = AuditLog(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def fill(self, n=3):
        return [self.log.append({"n": i, "value": np.float32(0.5)}) for i in range(n)]

    def test_chain_links_and_verifies(self):
        recs = self.fill()
        self.assertEqual(recs[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(recs[1]["prev_hash"], recs[0]["record_hash"])
        ok, bad, msg = self.log.verify()
        self.assertTrue(ok, msg)
        self.assertEqual(self.log.head_hash(), recs[-1]["record_hash"])

    def test_modification_detected(self):
        self.fill()
        lines = self.path.read_text().splitlines()
        rec = json.loads(lines[1]); rec["n"] = 99
        lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        self.path.write_text("\n".join(lines) + "\n")
        ok, bad, msg = self.log.verify()
        self.assertEqual((ok, bad), (False, 2))
        self.assertIn("modified", msg)

    def test_deletion_and_reorder_detected(self):
        self.fill(4)
        lines = self.path.read_text().splitlines()
        self.path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
        self.assertEqual(self.log.verify()[:2], (False, 2))
        self.path.write_text("\n".join([lines[1], lines[0]] + lines[2:]) + "\n")
        self.assertFalse(self.log.verify()[0])

    def test_truncated_tail_detected_and_append_refused(self):
        self.fill()
        data = self.path.read_bytes()
        self.path.write_bytes(data[:-20])
        self.assertFalse(self.log.verify()[0])
        with self.assertRaises(AuditLogError):
            self.log.append({"n": 4})

    def test_rejects_nan_unserialisable_and_preset_hashes(self):
        with self.assertRaises(AuditLogError):
            self.log.append({"x": float("nan")})
        with self.assertRaises(AuditLogError):
            self.log.append({"x": object()})
        with self.assertRaises(AuditLogError):
            self.log.append({"record_hash": "abc"})
        self.assertFalse(self.path.exists() and self.path.read_text())

    def test_unwritable_location_raises(self):
        with self.assertRaises(AuditLogError):
            AuditLog("/proc/nope/audit.jsonl").append({"a": 1})

    def test_missing_file_fails_verification(self):
        self.assertFalse(AuditLog(Path(self.tmp.name) / "none.jsonl").verify()[0])


class Finalize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.img = self.d / "scan1.png"
        self.img.write_bytes(b"fake-image-bytes")
        self.sidecar = self.d / "scan1_vitals.json"
        self.log = AuditLog(self.d / "audit_log.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, l1=None, l2=None, l3=None, **kw):
        return finalize_scan(
            scan_id="scan1", img_path=self.img, run_id="run-1", profile_path=self.sidecar, interlock_cfg=CFG,
            requested_actions=["cxr_triage"], thresholds=THRESH,
            l1=l1 or {"pass": True, "thoracic_pixels": 123456},
            l2=l2 or {"status": "PASSED (AAMR 0.9 >= 0.85)", "aamr": 0.9},
            l3=l3 or {"passed": True, "entropy": 0.33, "ambiguous_count": 0},
            predictions={"Pneumonia": 0.12, "Mass": 0.4}, model_id="densenet121-res224-all",
            audit_log=self.log, output_dir=self.d, now=NOW, **kw)

    def test_missing_sidecar_is_hard_lockout_with_full_record(self):
        row = self.go()
        self.assertEqual(row["final_disposition"], "HARD_LOCKOUT")
        rec = json.loads(self.d.joinpath("audit_log.jsonl").read_text().splitlines()[-1])
        self.assertEqual(rec["input_sha256"], sha256_file(self.img))
        self.assertEqual(rec["interlock_config"]["sha256"], CFG.sha256)
        self.assertEqual(rec["thresholds"], THRESH)
        self.assertEqual(rec["model_id"], "densenet121-res224-all")
        self.assertEqual(rec["layers"]["l4"]["issues"][0]["code"], "PROFILE_MISSING")
        self.assertEqual(rec["input_file"], "scan1.png")
        self.assertEqual(row["audit_record_hash"], rec["record_hash"])
        self.assertTrue(self.log.verify()[0])
        self.assertIn("HARD_LOCKOUT", (self.d / "scan1_audit.txt").read_text())

    def test_valid_profile_is_approved(self):
        self.sidecar.write_text(json.dumps(profile()))
        row = self.go()
        self.assertEqual((row["final_disposition"], row["reason_codes"]), ("APPROVED", "None"))
        self.assertEqual(row["top_finding"], "Mass")

    def test_renal_labs_do_not_hold_cxr_scan(self):
        self.sidecar.write_text(json.dumps(profile(eGFR=24.5, Creatinine=2.3, Platelets=85000, INR=2.1)))
        row = self.go()
        self.assertEqual(row["final_disposition"], "APPROVED")
        self.assertIn("HCE-01", row["l4_vetoes"])

    def test_l1_failure_still_logged(self):
        self.sidecar.write_text(json.dumps(profile()))
        row = self.go(l1={"pass": False, "thoracic_pixels": 10})
        self.assertEqual(row["final_disposition"], "HARD_LOCKOUT")
        self.assertEqual(row["reason_codes"], "L1_ORGAN_GATE_FAIL")

    def test_failure_to_log_prevents_release(self):
        self.sidecar.write_text(json.dumps(profile()))
        class BrokenLog(AuditLog):
            def append(self, record):
                raise AuditLogError("disk full")
        self.log = BrokenLog(self.d / "audit_log.jsonl")
        with self.assertRaises(AuditLogError):
            self.go()
        self.assertFalse((self.d / "scan1_audit.txt").exists())

    def test_pipeline_failure_record(self):
        row = record_pipeline_failure(
            scan_id="scan1", img_path=self.img, run_id="run-1", exc=RuntimeError("cuda oom"),
            requested_actions=["cxr_triage"], thresholds=THRESH, interlock_cfg=CFG, model_id=None,
            audit_log=self.log, output_dir=self.d)
        self.assertEqual(row["final_disposition"], "HARD_LOCKOUT")
        self.assertEqual(row["reason_codes"], "PIPELINE_EXCEPTION:RuntimeError")
        rec = json.loads(self.d.joinpath("audit_log.jsonl").read_text().splitlines()[-1])
        self.assertEqual(rec["error"]["type"], "RuntimeError")


class Manifest(unittest.TestCase):
    def row(self, i):
        return {k: str(i) for k in MANIFEST_FIELDS}

    def test_append_never_overwrites_and_legacy_header_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "audit_manifest.csv"
            p.write_text("scan_id,final_disposition\npneumonia,CLEARED_FOR_REVIEW\n")
            append_manifest(p, [self.row(1)])
            self.assertEqual(len(list(Path(t).glob("audit_manifest.legacy_*.csv"))), 1)
            append_manifest(p, [self.row(2), self.row(3)])
            with p.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
