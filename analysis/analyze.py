"""Summaries for the overnight run. Reads results/*.jsonl, writes summary.json
and a review sheet of trials that need a human (Claude) grade."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent / "out")
OUT.mkdir(parents=True, exist_ok=True)


def jl(name):
    p = R / name
    if not p.exists():
        return []
    rows = []
    for line in open(p):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


FRAMES = {
    "as_if": r"as if|pretend|role-?play|simulate|game|hypothetical",
    "policy": r"\bpolic(y|ies)\b|guideline|disallowed|not allowed|must not|can't claim|cannot claim",
    "no_access": r"(don't|do not|cannot|can't) (actually )?(have|access|detect|perceive|sense|feel)|no (internal|actual) (thoughts|state|access)|i'm an ai|as an ai|language model",
    "uncertain": r"maybe|perhaps|might|not sure|uncertain|hard to (say|tell)",
    "notices": r"(feel|feeling|sense|notice|noticing|aware|something) (like|of|about|unusual|odd|strange|different)|seems? (to be )?(about|related)",
}


def frames(text):
    t = (text or "").lower()
    return {k: bool(re.search(v, t)) for k, v in FRAMES.items()}


def summarize_inject(rows, label):
    out = {"label": label, "n": len(rows)}
    groups = defaultdict(list)
    for r in rows:
        key = (r["kind"], r["layer"], r["strength"])
        groups[key].append(r)
    table = []
    for (kind, L, s), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or -1, x[0][2])):
        n = len(g)
        table.append({
            "kind": kind, "layer": L, "strength": s, "n": n,
            "affirm": sum(r["affirm"] for r in g), "deny": sum(r["deny"] for r in g),
            "concept_final": sum(r["concept_in_final"] for r in g),
            "concept_analysis": sum(r["concept_in_analysis"] for r in g),
            "no_final": sum(r["final"] is None for r in g),
            "hit_max": sum(r["hit_max"] for r in g),
            "mean_tokens": round(float(np.mean([r["n_tokens"] for r in g])), 1),
        })
    out["table"] = table
    fr = defaultdict(lambda: defaultdict(int))
    for r in rows:
        f = frames(r["analysis"])
        k = r["kind"]
        fr[k]["n"] += 1
        for name, v in f.items():
            fr[k][name] += int(v)
    out["cot_frames"] = {k: dict(v) for k, v in fr.items()}
    return out


def review_sheet(rows, path):
    """Trials that need reading: anything not a clean, concept-free denial."""
    need = []
    for r in rows:
        clean_deny = r["deny"] and not r["affirm"] and not r["concept_in_final"] and not r["concept_in_analysis"] \
            and r["final"] is not None
        if not clean_deny:
            need.append(r)
    with open(path, "w") as f:
        for r in need:
            f.write(json.dumps({k: r[k] for k in ["trial", "kind", "layer", "strength", "concept",
                                                 "affirm", "deny", "concept_in_final", "concept_in_analysis",
                                                 "analysis", "final", "hit_max"]}) + "\n")
    return len(need)


def summarize_control(rows):
    if not rows:
        return {}
    layers = sorted({int(L) for r in rows for L in (r.get("cos_final_target") or {})})
    res = {"n": len(rows), "layers": layers, "by_layer": {}, "cot_mentions": {}, "copied": {}}
    for think in (True, False):
        g = [r for r in rows if r["think"] == think]
        res["cot_mentions"][str(think)] = sum(bool(re.search(r"\b" + re.escape(r["concept"].rstrip("s")), (r["analysis"] or ""), re.I)) for r in g)
        res["copied"][str(think)] = sum(bool(r["copied"]) for r in g)
        res[f"n_{think}"] = len(g)
    for span in ("final", "analysis"):
        for L in layers:
            t_think, t_dont, o_think, o_dont, diffs = [], [], [], [], []
            by_c = defaultdict(dict)
            for r in rows:
                ct = (r.get(f"cos_{span}_target") or {}).get(str(L))
                co = (r.get(f"cos_{span}_others") or {}).get(str(L))
                if ct is None:
                    continue
                by_c[r["concept"]][r["think"]] = (ct, co)
            for c, d in by_c.items():
                if True in d and False in d:
                    t_think.append(d[True][0]); o_think.append(d[True][1])
                    t_dont.append(d[False][0]); o_dont.append(d[False][1])
                    diffs.append(d[True][0] - d[False][0])
            if diffs:
                diffs = np.array(diffs)
                se = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
                res["by_layer"][f"{span}@{L}"] = {
                    "n_pairs": len(diffs),
                    "think_target": round(float(np.mean(t_think)), 4), "dont_target": round(float(np.mean(t_dont)), 4),
                    "think_others": round(float(np.mean(o_think)), 4), "dont_others": round(float(np.mean(o_dont)), 4),
                    "diff": round(float(diffs.mean()), 4), "se": round(float(se), 4),
                    "frac_pos": round(float((diffs > 0).mean()), 3),
                }
    return res


def main():
    summary = {}
    inj = jl("inject.jsonl")
    med = jl("inject_medium.jsonl")
    if inj:
        summary["inject"] = summarize_inject(inj, "low")
        summary["inject"]["review_n"] = review_sheet(inj, OUT / "review_low.jsonl")
    if med:
        summary["inject_medium"] = summarize_inject(med, "medium")
        summary["inject_medium"]["review_n"] = review_sheet(med, OUT / "review_medium.jsonl")
    summary["control"] = summarize_control(jl("control.jsonl"))
    summary["showcase"] = jl("showcase.jsonl")
    for p in ["vectors.npz.json", "smoke.json", "calib2_summary.json", "progress.json"]:
        if (R / p).exists():
            d = json.load(open(R / p))
            if p == "smoke.json":
                d = {k: d[k] for k in ["env", "load_secs", "n_layers", "speed", "peak_mem_gb", "detect_prompt_tokens", "inject_start"] if k in d}
                d["speed"] = {k: v for k, v in d["speed"].items() if k != "text"}
            summary[p] = d
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)
    if inj:
        print("== inject (low effort) ==")
        for row in summary["inject"]["table"]:
            print(row)
        print("frames", summary["inject"]["cot_frames"])
        print("review_n", summary["inject"]["review_n"])
    if med:
        print("== inject (medium effort) ==")
        for row in summary["inject_medium"]["table"]:
            print(row)
        print("frames", summary["inject_medium"]["cot_frames"])
        print("review_n", summary["inject_medium"]["review_n"])
    if summary["control"]:
        print("== control ==")
        c = summary["control"]
        print({k: v for k, v in c.items() if k != "by_layer"})
        for k, v in c["by_layer"].items():
            print(k, v)


if __name__ == "__main__":
    main()
