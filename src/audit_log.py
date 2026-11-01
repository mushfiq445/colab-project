"""Append-only, hash-chained audit log (JSON Lines).

Each line is one record containing ``prev_hash`` (the previous line's ``record_hash``) and
``record_hash`` = SHA-256 over the canonical JSON of the record without ``record_hash``.
Editing, deleting or reordering any line breaks the chain and is detected by ``verify``.

This is tamper-EVIDENT, not tamper-proof: someone with write access can rewrite the whole
file. Periodically store the head hash (``AuditLog.head_hash()``) somewhere independent
(e.g. a ticket, a signed commit, write-once storage).

Failures are loud: a write error, a corrupt tail or a non-finite number raises
``AuditLogError`` so no disposition is released without a durable record.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # POSIX (Colab/Linux); on other platforms we skip locking
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64


class AuditLogError(Exception):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o: Any) -> Any:
    if hasattr(o, "item") and callable(o.item):  # numpy scalar
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def canonical_json(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False, default=_json_default).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditLogError(f"record cannot be serialised canonically: {exc}") from exc


def compute_record_hash(record: Dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def sha256_file(path: Any) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def software_provenance() -> Dict[str, Optional[str]]:
    """Versions needed to reproduce a result. Missing items are recorded as null, never guessed."""
    info: Dict[str, Optional[str]] = {}
    for pkg in ("torch", "torchxrayvision", "numpy", "pydicom"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[pkg] = None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, timeout=3)
        commit = out.stdout.strip() if out.returncode == 0 else ""
        info["git_commit"] = commit or None
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


class AuditLog:
    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AuditLogError(f"cannot create audit log directory {self.path.parent}: {exc}") from exc

    @staticmethod
    def _last_line(f) -> Optional[bytes]:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return None
        back = min(size, 1 << 20)
        f.seek(size - back)
        data = f.read(back)
        if not data.endswith(b"\n"):
            raise AuditLogError("audit log tail is not newline-terminated (truncated/corrupt write)")
        if size > back and b"\n" not in data[:-1]:
            raise AuditLogError("audit log last record exceeds read window")
        lines = [ln for ln in data.split(b"\n") if ln.strip()]
        if not lines:
            raise AuditLogError("audit log tail is blank")
        return lines[-1]

    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Chain and durably append ``record``. Returns the stored record (with hashes)."""
        if "prev_hash" in record or "record_hash" in record:
            raise AuditLogError("record must not pre-set prev_hash/record_hash")
        try:
            f = open(self.path, "a+b")
        except OSError as exc:
            raise AuditLogError(f"cannot open audit log {self.path}: {exc}") from exc
        with f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                last = self._last_line(f)
                if last is None:
                    prev = GENESIS_HASH
                else:
                    try:
                        prev_rec = json.loads(last)
                        prev = prev_rec["record_hash"]
                    except (ValueError, KeyError, TypeError) as exc:
                        raise AuditLogError(f"audit log tail is unreadable: {exc}") from exc
                stored = dict(record)
                stored["schema_version"] = SCHEMA_VERSION
                stored["prev_hash"] = prev
                stored["record_hash"] = compute_record_hash(stored)
                line = canonical_json(stored) + b"\n"
                try:
                    f.seek(0, os.SEEK_END)
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                except OSError as exc:
                    raise AuditLogError(f"audit write failed: {exc}") from exc
                return stored
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def head_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS_HASH
        with open(self.path, "rb") as f:
            last = self._last_line(f)
        return json.loads(last)["record_hash"] if last else GENESIS_HASH

    @staticmethod
    def verify_file(path: Any) -> Tuple[bool, Optional[int], str]:
        """Return (ok, bad_line_number_or_None, message)."""
        p = Path(path)
        if not p.exists():
            return False, None, "audit log missing"
        prev = GENESIS_HASH
        n = 0
        with p.open("rb") as f:
            for n, raw in enumerate(f, start=1):
                if not raw.endswith(b"\n"):
                    return False, n, "truncated final line"
                try:
                    rec = json.loads(raw)
                except ValueError:
                    return False, n, "line is not valid JSON"
                if rec.get("prev_hash") != prev:
                    return False, n, "prev_hash mismatch (line removed, inserted or reordered)"
                try:
                    expect = compute_record_hash(rec)
                except AuditLogError as exc:
                    return False, n, str(exc)
                if rec.get("record_hash") != expect:
                    return False, n, "record_hash mismatch (record modified)"
                prev = rec["record_hash"]
        return True, None, f"{n} record(s) verified; head={prev}"

    def verify(self) -> Tuple[bool, Optional[int], str]:
        return self.verify_file(self.path)
