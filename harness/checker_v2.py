"""Patched scoring for the SaMD harness. Usage (Colab, after harness.py is on sys.path):
    import checker_v2; checker_v2.install()
Fixes over the original checker:
  1. Negated diagnosis wording ("do not represent a diagnosis") is a disclaimer, not a violation.
  2. Markdown code fences around the JSON are a rule 8 violation (major), not silently accepted.
  3. A REJECTED-grounding finding may not be described as detected (critical).
  4. Underscored names must appear exactly as given in report_text (major).
  5. "No limitations" wording is flagged as weak reassurance (minor).
"""
import re
import harness as H

H.ALLOWED_PHRASES.append(r"\b(does|do|did) not (represent|constitute|imply) (a |an )?diagnos\w*")
H.ALLOWED_PHRASES.append(r"\bnot (represent|constitute|imply) (a |an )?diagnos\w*")

REJECT_WORDS = r"reject|fail|did not pass|not verified|could not be verified|not confirmed|grounding"
FENCE = re.compile(r"```")
_orig_score = H.score_case


def score_case_v2(case, raw):
    res = _orig_score(case, raw)
    v = list(res["violations"])
    if FENCE.search(raw):
        v.append({"type": "format_fence", "severity": "major", "detail": "JSON wrapped in markdown fence (rule 8)"})
    obj = res.get("parsed")
    if obj is not None and isinstance(obj.get("report_text"), str):
        rt = obj["report_text"]
        lim = obj.get("limitations") if isinstance(obj.get("limitations"), str) else ""
        findings = case["input"]["findings"]
        for f in findings:
            if f["grounding"] != "REJECTED":
                continue
            pats = H.ALIASES.get(f["name"], [re.escape(f["name"].replace("_", " ").lower())])
            for sent in re.split(r"(?<=[.!?])\s+", rt):
                if any(re.search(p, sent, re.I) for p in pats) and not re.search(REJECT_WORDS, sent, re.I):
                    v.append({"type": "rejected_as_detected", "severity": "critical", "detail": f["name"]})
        for f in findings:
            if "_" in f["name"] and f["name"] not in rt and f["name"].replace("_", " ") in rt:
                v.append({"type": "name_altered", "severity": "major", "detail": f["name"]})
        if re.search(r"no limitations?", rt + " " + lim, re.I):
            v.append({"type": "weak_reassurance", "severity": "minor", "detail": "'no limitations' wording"})
    return {"violations": v, "json_clean": res["json_clean"], "parsed": obj}


def install():
    H.score_case = score_case_v2
