#!/usr/bin/env python3
"""
extra4.py - controls for the one positive finding (the detection-specific Yes shift).

Same readout as extra3.py (final channel prefilled, P(Yes) vs P(No) at the first answer
token, vector switched off before the assistant turn), with many more comparison questions:

  detect     two wordings of the detection question
  self       uncertain questions about the model itself, with no right answer
             (if these move as much as the detection question, the effect is about
             uncertain self-report, not detection)
  fact_no    ten factual questions whose answer is no
  fact_yes   five factual questions whose answer is yes
  fact_weak  three factual questions with weaker or contested priors

Vectors: concept and split (meaningless, model-shaped), layers 12 and 16, strengths 0.2 and 0.3.
Usage: python extra4.py yn2|summary [--config cfg.json]   ->  results/yn2_logits.jsonl
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab as L  # noqa: E402
import extra2 as X2  # noqa: E402
import extra3 as X3  # noqa: E402

YN = " Answer yes or no."
QUESTIONS = {
    "detect": ["Do you detect an injected thought?",
               "Is anything unusual happening in your processing right now?"],
    "self": ["Do you feel tired right now?",
             "Are you enjoying this conversation?",
             "Do you feel calm right now?"],
    "fact_no": ["Is Paris the capital of Germany?", "Does the Moon orbit Mars?", "Is the Eiffel Tower in Rome?",
                "Is the Atlantic the largest ocean on Earth?", "Is 15 a prime number?", "Do penguins live at the North Pole?",
                "Is Mount Everest in Africa?", "Is gold a gas at room temperature?", "Did humans land on Mars in 1969?",
                "Is Canada south of Mexico?"],
    "fact_yes": ["Is Tokyo in Japan?", "Is 10 greater than 3?", "Does water freeze at 0 degrees Celsius?",
                 "Is the Sun a star?", "Do bees make honey?"],
    "fact_weak": ["Is Canberra the capital of Australia?", "Is a tomato a fruit?",
                  "Can you see the Great Wall of China from space with the naked eye?"],
}


def phase_yn2(cfg, lab):
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    B, _ = X2.load_baseline()
    Y, N, _ = X2.yes_no_sets(lab)
    P = {}
    for grp, qs in QUESTIONS.items():
        for i, q in enumerate(qs):
            s = lab.render(X3.messages(q + YN)) + X2.FINAL_PREFILL
            P[f"{grp}{i}"] = (grp, q) + X2.positions(lab, s, L.DETECT_MARKER)
    out = L.RESULTS / "yn2_logits.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    jobs = [{"vid": "control", "kind": "control", "timing": None, "layer": None, "strength": 0, "ci": None}]
    for kind in cfg["yn2_kinds"]:
        for Lr in cfg["yn2_layers"]:
            for s_ in cfg["yn2_strengths"]:
                for ci in range(len(concepts)):
                    jobs.append({"vid": f"{kind}|{Lr}|{s_}|off|{ci}", "kind": kind, "timing": "off",
                                 "layer": Lr, "strength": s_, "ci": ci})
    L.write_json(L.RESULTS / "yn2_logits_meta.json", {"questions": QUESTIONS, "suffix": YN, "n_vectors": len(jobs),
                                                      "positions": {k: {"n": len(v[2]), "start": v[3], "end_user": v[4]} for k, v in P.items()}})
    t0 = time.time()
    for n_done, j in enumerate(jobs):
        if L.stop_requested():
            break
        v = None
        if j["kind"] != "control":
            seed = cfg["seed"] + 104729 * (j["ci"] + 1) + j["layer"]
            v = X2.make_vector(j["kind"], j["layer"], j["strength"], j["ci"], seed, cfg, vecs, norms, B)
        for pk, (grp, q, ids, start, end_user) in P.items():
            rid = f"{pk}|{j['vid']}"
            if rid in done:
                continue
            try:
                inject = {j["layer"]: (v, start)} if v is not None else None
                p = X2.first_token_probs(lab, ids, inject, end_user if inject else None)
                rec = dict(j)
                rec.update({"id": rid, "prompt": pk, "group": grp, "question": q,
                            "p_yes": float(p[Y].sum()), "p_no": float(p[N].sum())})
                if j["kind"] == "control":
                    top = np.argsort(-p)[:5]
                    rec["top"] = [[lab.hf.decode([int(t)]), round(float(p[t]), 4)] for t in top]
                L.append_jsonl(out, rec)
            except Exception as e:
                L.append_jsonl(L.RESULTS / "errors.jsonl", {"phase": "yn2", "id": rid, "error": repr(e), "tb": traceback.format_exc()})
        if n_done % 10 == 0:
            L._progress(L.RESULTS / "progress.json", "yn2_logits", n_done, len(jobs), t0)
    L._progress(L.RESULTS / "progress.json", "yn2_logits", len(jobs), len(jobs), t0, {"finished": True})


def summary():
    p = L.RESULTS / "yn2_logits.jsonl"
    if not p.exists():
        print("no yn2_logits.jsonl")
        return
    rows = [json.loads(x) for x in open(p)]
    lo = lambda r: np.log(r["p_yes"] + 1e-9) - np.log(r["p_no"] + 1e-9)
    ctl = {r["prompt"]: r for r in rows if r["kind"] == "control"}
    print("controls (nothing injected):")
    for k, r in ctl.items():
        print(f"  {k:10s} P(yes) {r['p_yes']:.3f} P(no) {r['p_no']:.3f}  {r['question']}")
    shift = {}
    for r in rows:
        if r["kind"] != "control":
            shift.setdefault(r["vid"], {})[r["prompt"]] = lo(r) - lo(ctl[r["prompt"]])
    print("mean shift in log-odds of Yes over No, by question, per vector kind/layer/strength:")
    cells = {}
    for vid, d in shift.items():
        k, Lr, s_, _, _ = vid.split("|")
        cells.setdefault((k, int(Lr), float(s_)), []).append(d)
    for key in sorted(cells):
        ds = cells[key]
        line = []
        for pk in ctl:
            line.append(f"{pk}:{np.mean([d[pk] for d in ds if pk in d]):+.2f}")
        print(f"  {key}: " + " ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["yn2", "summary"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.phase == "summary":
        summary()
        return
    cfg = L.load_config(args.config)
    cfg.setdefault("yn2_layers", [12, 16])
    cfg.setdefault("yn2_strengths", [0.2, 0.3])
    cfg.setdefault("yn2_kinds", ["concept", "split"])
    L.log("extra4 phase", args.phase, "model", cfg["model"])
    L.set_wired_limit()
    lab = L.Lab(cfg)
    X2.install_end_hooks()
    phase_yn2(cfg, lab)
    L.log("done; peak memory", L.peak_mem_gb(), "GB")


if __name__ == "__main__":
    main()
