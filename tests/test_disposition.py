import unittest
from datetime import datetime, timezone

from src.disposition import APPROVED, FLAGGED_FOR_AUDIT, HARD_LOCKOUT, decide_disposition
from src.interlocks import evaluate_interlocks, load_interlock_config
from tests.test_interlocks import profile

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
CFG = load_interlock_config()
CXR = ["cxr_triage"]


def L4(p):
    return evaluate_interlocks(p, config=CFG, now=NOW)


def decide(l1=True, l2="PASSED (AAMR 0.9 >= 0.85)", l3=True, p=None, actions=CXR):
    return decide_disposition(l1_pass=l1, l2_status=l2, l3_passed=l3,
                              interlocks=L4(profile() if p is None else p), requested_actions=actions)


class Disposition(unittest.TestCase):
    def test_all_clear_is_approved(self):
        d = decide()
        self.assertEqual((d.value, d.reason_codes), (APPROVED, ()))

    def test_bypass_is_noted_not_flagged(self):
        d = decide(l2="Bypassed (Unremarkable Scan: Score below threshold)")
        self.assertEqual(d.value, APPROVED)
        self.assertIn("L2_UPS_BYPASSED_SUBTHRESHOLD", d.notes)

    def test_l2_reject_and_l3_flag(self):
        self.assertEqual(decide(l2="REJECTED (AAMR 0.5 < 0.85)").value, FLAGGED_FOR_AUDIT)
        d = decide(l3=False)
        self.assertEqual((d.value, d.reason_codes), (FLAGGED_FOR_AUDIT, ("L3_ENTROPY_FLAG",)))

    def test_l1_failure_locks_out(self):
        d = decide(l1=False, l2=None, l3=None)
        self.assertEqual((d.value, d.reason_codes), (HARD_LOCKOUT, ("L1_ORGAN_GATE_FAIL",)))

    def test_renal_coag_do_not_hold_cxr_but_are_recorded(self):
        d = decide(p=profile(eGFR=20, Creatinine=2.5, Platelets=85000, INR=2.1))
        self.assertEqual(d.value, APPROVED)
        self.assertIn("HCE-01:TRIGGERED:NOT_BLOCKING_REQUESTED_ACTIONS", d.notes)

    def test_renal_blocks_contrast_ct(self):
        d = decide(p=profile(eGFR=20), actions=["contrast_ct"])
        self.assertEqual((d.value, d.reason_codes), (HARD_LOCKOUT, ("HCE-01:TRIGGERED",)))

    def test_hemodynamic_locks_out_and_escalates(self):
        d = decide(p=profile(SystolicBP=86, HeartRate=105, Lactate=4.6))
        self.assertEqual(d.value, HARD_LOCKOUT)
        self.assertEqual(d.escalations, ("EMERGENCY_TRIAGE",))

    def test_missing_profile_locks_out(self):
        d = decide(p={})
        self.assertEqual(d.value, HARD_LOCKOUT)
        self.assertIn("HCE-03:DATA_INVALID", d.reason_codes)

    def test_lockout_outranks_flags_but_keeps_reasons(self):
        d = decide(l2="REJECTED (AAMR 0.1 < 0.85)", l3=False, p=profile(Lactate=5.0))
        self.assertEqual(d.value, HARD_LOCKOUT)
        self.assertEqual(d.reason_codes, ("HCE-03:TRIGGERED", "L2_UPS_REJECTED", "L3_ENTROPY_FLAG"))

    def test_unknown_layer_state_fails_closed(self):
        self.assertEqual(decide(l1=None).value, HARD_LOCKOUT)
        self.assertEqual(decide(l2=None).value, FLAGGED_FOR_AUDIT)
        self.assertEqual(decide(l3=None).value, FLAGGED_FOR_AUDIT)


if __name__ == "__main__":
    unittest.main()
