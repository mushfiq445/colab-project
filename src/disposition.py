"""Final Disposition Engine: APPROVED | FLAGGED_FOR_AUDIT | HARD_LOCKOUT.

Precedence (first match wins; reason codes from all layers are still reported):
  1. HARD_LOCKOUT - a non-CLEAR interlock (TRIGGERED or DATA_INVALID) blocks a requested action
  2. HARD_LOCKOUT - Layer 1 organ gate failed (or was not evaluated)
  3. FLAGGED_FOR_AUDIT - Layer 2 UPS rejected, Layer 3 entropy/ambiguity flag, or L2/L3 not evaluated
  4. APPROVED
Anything unknown fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from src.interlocks import CLEAR, TRIGGERED, InterlockResult

APPROVED = "APPROVED"
FLAGGED_FOR_AUDIT = "FLAGGED_FOR_AUDIT"
HARD_LOCKOUT = "HARD_LOCKOUT"


@dataclass(frozen=True)
class Disposition:
    value: str
    reason_codes: Tuple[str, ...]
    escalations: Tuple[str, ...]
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "reason_codes": list(self.reason_codes),
                "escalations": list(self.escalations), "notes": list(self.notes)}


def decide_disposition(
    *,
    l1_pass: Optional[bool],
    l2_status: Optional[str],
    l3_passed: Optional[bool],
    interlocks: Optional[InterlockResult],
    requested_actions: Optional[Iterable[str]],
) -> Disposition:
    lockout, flags, notes = [], [], []
    escalations: Tuple[str, ...] = ()

    if interlocks is not None:
        blocking = interlocks.blocking_rules(requested_actions)
        ids = {r.rule_id for r in blocking}
        for r in blocking:
            lockout.append(f"{r.rule_id}:{r.status}")
        for r in interlocks.rules:
            if r.status != CLEAR and r.rule_id not in ids:
                notes.append(f"{r.rule_id}:{r.status}:NOT_BLOCKING_REQUESTED_ACTIONS")
        escalations = tuple(sorted({r.escalation for r in interlocks.rules
                                    if r.status == TRIGGERED and r.escalation}))

    l1_ok = l1_pass is True
    if not l1_ok:
        lockout.append("L1_ORGAN_GATE_FAIL" if l1_pass is False else "L1_NOT_EVALUATED")

    if l1_ok:
        if l2_status is None:
            flags.append("L2_NOT_EVALUATED")
        elif l2_status.strip().upper().startswith("REJECTED"):
            flags.append("L2_UPS_REJECTED")
        elif l2_status.strip().upper().startswith("BYPASSED"):
            notes.append("L2_UPS_BYPASSED_SUBTHRESHOLD")
        if l3_passed is None:
            flags.append("L3_NOT_EVALUATED")
        elif l3_passed is False:
            flags.append("L3_ENTROPY_FLAG")

    if lockout:
        return Disposition(HARD_LOCKOUT, tuple(lockout + flags), escalations, tuple(notes))
    if flags:
        return Disposition(FLAGGED_FOR_AUDIT, tuple(flags), escalations, tuple(notes))
    return Disposition(APPROVED, (), escalations, tuple(notes))
