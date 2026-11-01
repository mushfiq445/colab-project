"""Per-scan finalisation: Layer 4 interlocks -> disposition -> durable audit record.

Torch-free on purpose so the whole decision/audit path is unit-testable.
Every scan - including ones rejected at Layer 1 or that crashed - gets exactly one chained
audit record, and no result is released if that record cannot be written.
"""
from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.audit_log import AuditLog, sha256_file, software_provenance, utc_now_iso
from src.disposition import HARD_LOCKOUT, Disposition, decide_disposition
from src.interlocks import InterlockConfig, InterlockResult, evaluate_interlocks, load_clinical_profile

MANIFEST_FIELDS = [
    "run_id", "timestamp_utc", "scan_id", "l1_organ_pass", "thoracic_pixels", "top_finding",
    "top_score", "l2_ups_status", "aamr", "l3_entropy", "l3_ambiguous_count", "l4_vetoes",
    "l4_data_issues", "final_disposition", "reason_codes", "audit_record_hash",
]


def _row(scan_id, run_id, ts, l1, l2, l3, predictions, interlocks, disp, record_hash) -> Dict[str, Any]:
    top, top_score = ("None", 0.0)
    if predictions:
        top, top_score = max(predictions.items(), key=lambda kv: kv[1])
    return {
        "run_id": run_id, "timestamp_utc": ts, "scan_id": scan_id,
        "l1_organ_pass": (l1 or {}).get("pass", False), "thoracic_pixels": (l1 or {}).get("thoracic_pixels", 0),
        "top_finding": top, "top_score": round(float(top_score), 4),
        "l2_ups_status": (l2 or {}).get("status", "N/A"), "aamr": round(float((l2 or {}).get("aamr", 0.0)), 3),
        "l3_entropy": round(float((l3 or {}).get("entropy", 0.0)), 4),
        "l3_ambiguous_count": (l3 or {}).get("ambiguous_count", 0),
        "l4_vetoes": "; ".join(interlocks.legacy_messages()) if interlocks and interlocks.legacy_messages() else "None",
        "l4_data_issues": "; ".join(f"{i.code}:{i.field or '-'}" for i in interlocks.issues) if interlocks and interlocks.issues else "None",
        "final_disposition": disp.value, "reason_codes": "; ".join(disp.reason_codes) or "None",
        "audit_record_hash": record_hash,
    }


def _write_report(output_dir: Any, scan_id: str, row: Mapping[str, Any], disp: Disposition,
                  interlocks: Optional[InterlockResult], error: Optional[Mapping[str, str]]) -> Path:
    path = Path(output_dir) / f"{scan_id}_audit.txt"
    lines = [f"SaMD Clinical Safety Audit: {scan_id}", "=" * 45,
             f"Disposition: {disp.value}",
             f"Reason codes: {', '.join(disp.reason_codes) or 'None'}",
             f"Escalations: {', '.join(disp.escalations) or 'None'}",
             f"Top Finding: {row['top_finding']} ({row['top_score']:.4f})",
             f"Layer 1 Organ Pass: {row['l1_organ_pass']} ({row['thoracic_pixels']} px)",
             f"Layer 2 UPS Status: {row['l2_ups_status']} (AAMR: {row['aamr']:.3f})",
             f"Layer 3 Entropy: {row['l3_entropy']:.4f} bits (Ambiguous: {row['l3_ambiguous_count']})",
             "Layer 4 Interlocks:"]
    if interlocks is None:
        lines.append("  (not evaluated)")
    else:
        for r in interlocks.rules:
            lines.append(f"  * [{r.rule_id}] {r.status}" + (f": {'; '.join(r.reasons)}" if r.reasons else ""))
        for i in interlocks.issues:
            lines.append(f"  ! data issue {i.code} ({i.field or '-'}): {i.message}")
    if error:
        lines.append(f"Pipeline error: {error['type']}: {error['message']}")
    lines.append(f"Audit record hash: {row['audit_record_hash']}")
    path.write_text("\n".join(lines) + "\n")
    return path


def finalize_scan(
    *,
    scan_id: str,
    img_path: Any,
    run_id: str,
    profile_path: Any,
    interlock_cfg: InterlockConfig,
    requested_actions: Iterable[str],
    thresholds: Mapping[str, Any],
    l1: Optional[Mapping[str, Any]],
    l2: Optional[Mapping[str, Any]],
    l3: Optional[Mapping[str, Any]],
    predictions: Optional[Mapping[str, float]],
    model_id: Optional[str],
    audit_log: AuditLog,
    output_dir: Any,
    now: Optional[datetime] = None,
    error: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate L4, decide disposition, write the report and chained record. Returns manifest row."""
    requested = tuple(requested_actions)
    profile, load_issue = load_clinical_profile(profile_path)
    interlocks = evaluate_interlocks(profile, config=interlock_cfg, now=now, load_issue=load_issue)
    disp = decide_disposition(
        l1_pass=None if l1 is None else l1.get("pass"),
        l2_status=None if l2 is None else l2.get("status"),
        l3_passed=None if l3 is None else l3.get("passed"),
        interlocks=interlocks, requested_actions=requested,
    )
    return _store(scan_id, img_path, run_id, requested, thresholds, interlock_cfg, l1, l2, l3, predictions,
                  model_id, audit_log, output_dir, interlocks, disp, error)


def record_pipeline_failure(
    *, scan_id: str, img_path: Any, run_id: str, exc: BaseException, requested_actions: Iterable[str],
    thresholds: Mapping[str, Any], interlock_cfg: InterlockConfig, model_id: Optional[str],
    audit_log: AuditLog, output_dir: Any,
) -> Dict[str, Any]:
    """A scan whose processing raised is locked out and logged; it is never silently dropped."""
    disp = Disposition(HARD_LOCKOUT, (f"PIPELINE_EXCEPTION:{type(exc).__name__}",), (), ())
    err = {"type": type(exc).__name__, "message": str(exc)[:500]}
    return _store(scan_id, img_path, run_id, tuple(requested_actions), thresholds, interlock_cfg,
                  None, None, None, None, model_id, audit_log, output_dir, None, disp, err)


def _store(scan_id, img_path, run_id, requested, thresholds, cfg, l1, l2, l3, predictions, model_id,
           audit_log, output_dir, interlocks, disp, error) -> Dict[str, Any]:
    ts = utc_now_iso()
    record = {
        "record_type": "scan_disposition", "event_id": str(uuid.uuid4()), "timestamp_utc": ts, "run_id": run_id,
        "scan_id": scan_id, "input_file": os.path.basename(str(img_path)), "input_sha256": sha256_file(img_path),
        "software": software_provenance(), "model_id": model_id,
        "interlock_config": {"file": cfg.path.name, "sha256": cfg.sha256},
        "thresholds": dict(thresholds), "requested_actions": list(requested),
        "layers": {"l1": None if l1 is None else dict(l1), "l2": None if l2 is None else dict(l2),
                   "l3": None if l3 is None else dict(l3),
                   "l4": None if interlocks is None else interlocks.to_dict()},
        "predictions": None if predictions is None else {k: float(v) for k, v in predictions.items()},
        "disposition": disp.to_dict(), "error": None if error is None else dict(error),
    }
    stored = audit_log.append(record)  # raises AuditLogError -> nothing is released
    row = _row(scan_id, run_id, ts, l1, l2, l3, predictions, interlocks, disp, stored["record_hash"])
    _write_report(output_dir, scan_id, row, disp, interlocks, error)
    return row


def append_manifest(csv_path: Any, rows: List[Dict[str, Any]]) -> None:
    """Append rows to the CSV view (the JSONL chain is authoritative). Never overwrites history."""
    p = Path(csv_path)
    if p.exists():
        with p.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != MANIFEST_FIELDS:
            legacy = p.with_name(f"{p.stem}.legacy_{datetime.now().strftime('%Y%m%dT%H%M%S')}{p.suffix}")
            p.rename(legacy)
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
