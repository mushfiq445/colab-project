"""Layer 4 - Hardcoded Clinical Interlocks (HCE), fail-closed.

Design rules
------------
* Thresholds, units, plausibility limits and freshness windows come from
  ``config/clinical_interlocks.yaml`` (single source of truth). Nothing is defaulted in code.
* Missing, unreadable, stale, non-numeric, non-finite, out-of-range or unit-unknown data is
  never replaced by a "healthy" value. A rule that cannot be evaluated is ``DATA_INVALID`` and
  is treated as blocking.
* Rules gate *actions* (``contrast_ct``, ``tpa_recommendation``, ``*``), not scans. A pipeline
  declares the actions it is about to perform; only rules blocking those actions lock it out.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import operator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "clinical_interlocks.yaml"
SECTIONS = ("vitals", "labs")
TRIGGERED, CLEAR, DATA_INVALID = "TRIGGERED", "CLEAR", "DATA_INVALID"
_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
# derived field -> (required inputs, function)
_DERIVED = {"ShockIndex": (("HeartRate", "SystolicBP"), lambda hr, sbp: hr / sbp)}


class ConfigError(Exception):
    """The interlock configuration is missing or invalid. Never recover with defaults."""


@dataclass(frozen=True)
class DataIssue:
    code: str
    field: Optional[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "field": self.field, "message": self.message}


@dataclass(frozen=True)
class InterlockConfig:
    path: Path
    sha256: str
    data: Dict[str, Any]


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    name: str
    status: str  # TRIGGERED | CLEAR | DATA_INVALID
    reasons: Tuple[str, ...]
    blocks: Tuple[str, ...]
    escalation: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"rule_id": self.rule_id, "name": self.name, "status": self.status,
                "reasons": list(self.reasons), "blocks": list(self.blocks), "escalation": self.escalation}


@dataclass(frozen=True)
class InterlockResult:
    rules: Tuple[RuleResult, ...]
    issues: Tuple[DataIssue, ...]
    warnings: Tuple[str, ...]
    values: Dict[str, Dict[str, Any]]
    config_sha256: str
    evaluated_at: str

    def blocking_rules(self, requested_actions: Optional[Iterable[str]] = None) -> List[RuleResult]:
        """Non-CLEAR rules that block any of ``requested_actions`` (None = every action)."""
        req = ("*",) if requested_actions is None else tuple(requested_actions)
        if not req:
            raise ValueError("requested_actions must not be empty")
        out = []
        for r in self.rules:
            if r.status == CLEAR:
                continue
            if "*" in req or "*" in r.blocks or set(r.blocks) & set(req):
                out.append(r)
        return out

    def legacy_messages(self) -> List[str]:
        msgs = []
        for r in self.rules:
            if r.status == TRIGGERED:
                msgs.append(f"[{r.rule_id}] {r.name}: " + "; ".join(r.reasons))
            elif r.status == DATA_INVALID:
                msgs.append(f"[{r.rule_id}] {r.name} CANNOT BE EVALUATED (DATA_INVALID): " + "; ".join(r.reasons))
        return msgs

    def to_dict(self) -> Dict[str, Any]:
        return {"config_sha256": self.config_sha256, "evaluated_at": self.evaluated_at,
                "rules": [r.to_dict() for r in self.rules],
                "issues": [i.to_dict() for i in self.issues],
                "warnings": list(self.warnings), "values": self.values}


# --------------------------------------------------------------------------- config

def _num(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x))


def _norm_unit(u: str) -> str:
    return u.strip().lower().replace("\u00b5", "u").replace("\u03bc", "u").replace(" ", "")


def _validate_config(d: Any) -> None:
    def bad(msg: str) -> None:
        raise ConfigError(f"clinical_interlocks.yaml: {msg}")

    if not isinstance(d, dict):
        bad("top level must be a mapping")
    if d.get("schema_version") != 1:
        bad("schema_version must be 1")
    pol = d.get("policy")
    if not isinstance(pol, dict):
        bad("missing 'policy'")
    for k in ("require_timestamps", "allow_unitless_values"):
        if not isinstance(pol.get(k), bool):
            bad(f"policy.{k} must be true/false")
    ma = pol.get("max_age_minutes")
    if not isinstance(ma, dict) or not all(_num(ma.get(s)) and ma[s] > 0 for s in SECTIONS):
        bad("policy.max_age_minutes needs a positive number for 'vitals' and 'labs'")
    if not _num(pol.get("future_skew_minutes")) or pol["future_skew_minutes"] < 0:
        bad("policy.future_skew_minutes must be a number >= 0")
    fields = d.get("fields")
    if not isinstance(fields, dict) or not fields:
        bad("missing 'fields'")
    for name, spec in fields.items():
        if not isinstance(spec, dict) or spec.get("section") not in SECTIONS:
            bad(f"fields.{name}.section must be one of {SECTIONS}")
        if not isinstance(spec.get("unit"), str) or not spec["unit"].strip():
            bad(f"fields.{name}.unit missing")
        if not (_num(spec.get("min")) and _num(spec.get("max")) and spec["min"] < spec["max"]):
            bad(f"fields.{name}.min/max must be numbers with min < max")
        conv = spec.get("conversions", {})
        if not isinstance(conv, dict) or not all(isinstance(k, str) and _num(v) and v > 0 for k, v in conv.items()):
            bad(f"fields.{name}.conversions must map unit strings to positive factors")
    for name, (req, _fn) in _DERIVED.items():
        if any(r not in fields for r in req):
            bad(f"derived field {name} needs fields {req}")
    rules = d.get("rules")
    if not isinstance(rules, dict) or not rules:
        bad("missing 'rules'")
    for rid, rule in rules.items():
        if not isinstance(rule, dict) or not isinstance(rule.get("name"), str):
            bad(f"rules.{rid}.name missing")
        blocks = rule.get("blocks")
        if not isinstance(blocks, list) or not blocks or not all(isinstance(b, str) and b for b in blocks):
            bad(f"rules.{rid}.blocks must be a non-empty list of action names")
        if rule.get("escalation") is not None and not isinstance(rule["escalation"], str):
            bad(f"rules.{rid}.escalation must be a string")
        crit = rule.get("criteria")
        if not isinstance(crit, list) or not crit:
            bad(f"rules.{rid}.criteria must be a non-empty list")
        for c in crit:
            if not isinstance(c, dict) or c.get("op") not in _OPS or not _num(c.get("value")):
                bad(f"rules.{rid} has a criterion with an invalid op/value: {c!r}")
            if c.get("field") not in fields and c.get("field") not in _DERIVED:
                bad(f"rules.{rid} references unknown field {c.get('field')!r}")


def load_interlock_config(path: Optional[Path] = None) -> InterlockConfig:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read interlock config {p}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"interlock config {p} is not valid YAML: {exc}") from exc
    _validate_config(data)
    return InterlockConfig(path=p, sha256=hashlib.sha256(raw).hexdigest(), data=data)


# --------------------------------------------------------------------------- profile input

def load_clinical_profile(path: Any) -> Tuple[Optional[Dict[str, Any]], Optional[DataIssue]]:
    """Read a sidecar JSON. Returns (profile, None) or (None, issue). Never substitutes defaults."""
    p = Path(path)
    if not p.exists():
        return None, DataIssue("PROFILE_MISSING", None, f"clinical profile not found: {p.name}")
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return None, DataIssue("PROFILE_UNREADABLE", None, f"{p.name}: {exc}")
    if not isinstance(data, dict):
        return None, DataIssue("PROFILE_UNREADABLE", None, f"{p.name}: top level must be an object")
    return data, None


def _parse_ts(raw: Any) -> Tuple[Optional[datetime], Optional[str]]:
    if not isinstance(raw, str) or not raw.strip():
        return None, "TIMESTAMP_INVALID"
    s = raw.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None, "TIMESTAMP_INVALID"
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return None, "TIMESTAMP_NAIVE"
    return dt.astimezone(timezone.utc), None


# --------------------------------------------------------------------------- evaluation

def evaluate_interlocks(
    profile: Optional[Mapping[str, Any]],
    *,
    config: InterlockConfig,
    now: Optional[datetime] = None,
    load_issue: Optional[DataIssue] = None,
) -> InterlockResult:
    cfg = config.data
    pol = cfg["policy"]
    now_utc = (now or datetime.now(timezone.utc))
    if now_utc.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now_utc.astimezone(timezone.utc)

    issues: List[DataIssue] = []
    warnings: List[str] = []
    seen: set = set()
    resolved: Dict[str, Optional[float]] = {}
    used: Dict[str, Dict[str, Any]] = {}

    def issue(code: str, fld: Optional[str], msg: str) -> None:
        if (code, fld) not in seen:
            seen.add((code, fld))
            issues.append(DataIssue(code, fld, msg))

    profile_ok = isinstance(profile, Mapping)
    if not profile_ok:
        li = load_issue or DataIssue("PROFILE_MISSING", None, "no clinical profile supplied")
        issue(li.code, li.field, li.message)

    def resolve(name: str) -> Optional[float]:
        if name in resolved:
            return resolved[name]
        resolved[name] = None  # fail closed unless fully validated below
        if not profile_ok:
            return None
        spec = cfg["fields"][name]
        section = spec["section"]
        sec = profile.get(section)
        if not isinstance(sec, Mapping):
            issue("SECTION_MISSING", name, f"profile section '{section}' missing")
            return None
        raw = sec.get(name)
        if raw is None:
            issue("FIELD_MISSING", name, f"{section}.{name} missing")
            return None
        if isinstance(raw, Mapping):
            val, unit, ts_raw = raw.get("value"), raw.get("unit"), raw.get("collected_at", sec.get("collected_at"))
        else:
            val, unit, ts_raw = raw, None, sec.get("collected_at")
        if not _num(val):
            code = "FIELD_NOT_FINITE" if isinstance(val, numbers.Real) and not isinstance(val, bool) else "FIELD_NOT_NUMERIC"
            issue(code, name, f"{name} value {val!r} is not a finite number")
            return None
        factor = 1.0
        shown_unit = spec["unit"]
        if unit is None:
            if not pol["allow_unitless_values"]:
                issue("UNIT_MISSING", name, f"{name} has no unit")
                return None
            warnings.append(f"UNITLESS_ASSUMED:{name} (assumed {spec['unit']})")
        else:
            table = {_norm_unit(spec["unit"]): 1.0}
            table.update({_norm_unit(k): float(v) for k, v in spec.get("conversions", {}).items()})
            key = _norm_unit(unit) if isinstance(unit, str) else None
            if key is None or key not in table:
                issue("UNIT_UNKNOWN", name, f"{name} unit {unit!r} not recognised (expected {spec['unit']})")
                return None
            factor = table[key]
        value = float(val) * factor
        if not (spec["min"] <= value <= spec["max"]):
            issue("OUT_OF_RANGE", name, f"{name} {value:g} {spec['unit']} outside plausible range "
                                        f"[{spec['min']:g}, {spec['max']:g}] (possible unit or entry error)")
            return None
        if ts_raw is None:
            if pol["require_timestamps"]:
                issue("TIMESTAMP_MISSING", name, f"{name} has no collected_at")
                return None
        else:
            ts, err = _parse_ts(ts_raw)
            if ts is None:
                issue(err or "TIMESTAMP_INVALID", name, f"{name} collected_at {ts_raw!r} invalid (need timezone-aware ISO 8601)")
                return None
            age = now_utc - ts
            if age > timedelta(minutes=pol["max_age_minutes"][section]):
                issue("STALE", name, f"{name} is {age} old (limit {pol['max_age_minutes'][section]} min)")
                return None
            if -age > timedelta(minutes=pol["future_skew_minutes"]):
                issue("FUTURE_TIMESTAMP", name, f"{name} collected_at is in the future")
                return None
        resolved[name] = value
        used[name] = {"value": value, "unit": shown_unit, "as_supplied": val,
                      "unit_supplied": unit if isinstance(unit, str) else None}
        return value

    def resolve_any(name: str) -> Optional[float]:
        if name in _DERIVED:
            req, fn = _DERIVED[name]
            vals = [resolve(r) for r in req]
            if any(v is None for v in vals):
                return None
            v = fn(*vals)
            used[name] = {"value": v, "unit": "ratio", "derived_from": list(req)}
            return v
        return resolve(name)

    results: List[RuleResult] = []
    for rid, rule in cfg["rules"].items():
        triggered: List[str] = []
        unavailable: List[str] = []
        for c in rule["criteria"]:
            v = resolve_any(c["field"])
            if v is None:
                unavailable.append(f"{c['field']} unavailable")
            elif _OPS[c["op"]](v, c["value"]):
                unit = cfg["fields"].get(c["field"], {}).get("unit", "ratio")
                triggered.append(f"{c['field']} {v:g} {unit} {c['op']} {c['value']:g}")
        if triggered:
            status, reasons = TRIGGERED, tuple(triggered)
        elif unavailable:
            status, reasons = DATA_INVALID, tuple(unavailable)
        else:
            status, reasons = CLEAR, ()
        results.append(RuleResult(rid, rule["name"], status, reasons, tuple(rule["blocks"]), rule.get("escalation")))

    return InterlockResult(
        rules=tuple(results), issues=tuple(issues), warnings=tuple(warnings), values=used,
        config_sha256=config.sha256, evaluated_at=now_utc.isoformat(),
    )
