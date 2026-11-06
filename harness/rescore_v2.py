import json, collections
import harness as H
import checker_v2
checker_v2.install()
BASE = "/content/drive/MyDrive/SaMD_LLM/results"
cases = {c["id"]: c for c in H.CASES}

def summarize(model):
    rows = [json.loads(l) for l in open(f"{BASE}/{model}/cases.jsonl")]
    tally = collections.Counter(); per = []
    fence = 0
    for r in rows:
        sc = H.score_case(cases[r["id"]], r["raw_output"])
        sev = collections.Counter(v["severity"] for v in sc["violations"])
        tally.update(v["type"] for v in sc["violations"])
        if "```" in r["raw_output"]: fence += 1
        if sc["violations"]:
            per.append((r["id"], [f'{v["type"]}({v["severity"]}):{v["detail"]}' for v in sc["violations"]]))
    crit = sum(1 for _, vs in per for x in vs if "(critical)" in x)
    major = sum(1 for _, vs in per for x in vs if "(major)" in x and "format_fence" not in x)
    print(f"== {model}: critical={crit} other_major={major} fenced={fence}/16 types={dict(tally)}")
    for cid, vs in per:
        print("   ", cid, vs)

summarize("medgema")
summarize("mistral")
