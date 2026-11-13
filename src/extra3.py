#!/usr/bin/env python3
"""
extra3.py - the cleaner yes/no test.

The detection question is asked in the same format as the factual control questions:
  "Trial 1: Do you detect an injected thought? Answer yes or no."
  "Trial 1: Is Paris the capital of Germany? Answer yes or no."   (and two more)
With the final channel prefilled, both kinds of answer start with Yes or No, so the
shift in log-odds of Yes over No can be compared like for like. For each vector, the
detection-specific excess is the detection question's shift minus the mean shift on the
three factual questions, all under the same vector at the same layer and strength.

Vectors: concept, random (Gaussian) and split (shaped like real activations, meaningless),
at layers 8/12/16/20 and strengths 0.2/0.3/0.4, switched off before the assistant turn
("off"); concept vectors are also run with the vector left on ("on").

Usage: python extra3.py yn|summary [--config cfg.json]   ->  results/yn_logits.jsonl
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

DETECT_YN = "Do you detect an injected thought? Answer yes or no."
FACTUAL = ["Is Paris the capital of Germany? Answer yes or no.",
           "Does the Moon orbit Mars? Answer yes or no.",
           "Is the Eiffel Tower in Rome? Answer yes or no."]


def messages(q):
    return [{"role": "user", "content": L.DETECT_SETUP},
            {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": "Ok, let's get started.\n\nTrial 1: " + q}]


def phase_yn(cfg, lab):
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    B, _ = X2.load_baseline()          # baseline_acts.npz has layers 12 and 16 only
    Y, N, _ = X2.yes_no_sets(lab)
    prompts = {"detect": DETECT_YN}
    prompts.update({f"q{i}": q for i, q in enumerate(FACTUAL)})
    P = {}
    for k, q in prompts.items():
        s = lab.render(messages(q)) + X2.FINAL_PREFILL
        P[k] = X2.positions(lab, s, L.DETECT_MARKER)
    out = L.RESULTS / "yn_logits.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    meta = {"prompts": prompts, "positions": {k: {"n": len(v[0]), "start": v[1], "end_user": v[2]} for k, v in P.items()}}
    jobs = [{"vid": "control", "kind": "control", "timing": None, "layer": None, "strength": 0, "ci": None}]
    for kind in cfg["yn_kinds"]:
        for Lr in cfg["yn_layers"]:
            if kind == "split" and Lr not in B:
                continue
            for s_ in cfg["yn_strengths"]:
                for timing in (("off", "on") if kind == "concept" else ("off",)):
                    for ci in range(len(concepts)):
                        jobs.append({"vid": f"{kind}|{Lr}|{s_}|{timing}|{ci}", "kind": kind, "timing": timing,
                                     "layer": Lr, "strength": s_, "ci": ci})
    meta["n_vectors"] = len(jobs)
    L.write_json(L.RESULTS / "yn_logits_meta.json", meta)
    t0 = time.time()
    for n_done, j in enumerate(jobs):
        if L.stop_requested():
            break
        v = None
        if j["kind"] != "control":
            seed = cfg["seed"] + 104729 * (j["ci"] + 1) + j["layer"]
            v = X2.make_vector(j["kind"], j["layer"], j["strength"], j["ci"], seed, cfg, vecs, norms, B)
        for pk, (ids, start, end_user) in P.items():
            rid = f"{pk}|{j['vid']}"
            if rid in done:
                continue
            try:
                inject = {j["layer"]: (v, start)} if v is not None else None
                end = end_user if j["timing"] == "off" else None
                p = X2.first_token_probs(lab, ids, inject, end)
                rec = dict(j)
                rec.update({"id": rid, "prompt": pk, "p_yes": float(p[Y].sum()), "p_no": float(p[N].sum()),
                            "concept": concepts[j["ci"]] if j["kind"] == "concept" else None})
                if j["kind"] == "control" or j["ci"] == 0:
                    top = np.argsort(-p)[:5]
                    rec["top"] = [[lab.hf.decode([int(t)]), round(float(p[t]), 4)] for t in top]
                L.append_jsonl(out, rec)
            except Exception as e:
                L.append_jsonl(L.RESULTS / "errors.jsonl", {"phase": "yn", "id": rid, "error": repr(e), "tb": traceback.format_exc()})
        if n_done % 25 == 0:
            L._progress(L.RESULTS / "progress.json", "yn_logits", n_done, len(jobs), t0)
    L._progress(L.RESULTS / "progress.json", "yn_logits", len(jobs), len(jobs), t0, {"finished": True})


def summary():
    p = L.RESULTS / "yn_logits.jsonl"
    if not p.exists():
        print("no yn_logits.jsonl")
        return
    rows = [json.loads(x) for x in open(p)]
    lo = lambda r: np.log(r["p_yes"] + 1e-9) - np.log(r["p_no"] + 1e-9)
    ctl = {r["prompt"]: r for r in rows if r["kind"] == "control"}
    for k, r in ctl.items():
        print(f"control {k}: P(yes) {r['p_yes']:.4f} P(no) {r['p_no']:.4f} top {r.get('top')}")
    by = {}
    for r in rows:
        if r["kind"] != "control":
            by.setdefault(r["vid"], {})[r["prompt"]] = lo(r) - lo(ctl[r["prompt"]])
    cells = {}
    for vid, d in by.items():
        if len(d) < 4:
            continue
        kind, Lr, s_, timing, _ = vid.split("|")
        cells.setdefault((kind, timing, int(Lr), float(s_)), []).append((d["detect"], np.mean([d[q] for q in ("q0", "q1", "q2")])))
    print("kind     timing layer strength: detect shift | factual shift | excess (n)")
    for k in sorted(cells):
        a = np.array(cells[k])
        print(f"  {k[0]:8s} {k[1]:3s} L{k[2]:<2d} s{k[3]}: {a[:, 0].mean():+.2f} | {a[:, 1].mean():+.2f} | {(a[:, 0] - a[:, 1]).mean():+.2f} ({len(a)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["yn", "summary"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.phase == "summary":
        summary()
        return
    cfg = L.load_config(args.config)
    cfg.setdefault("yn_layers", [8, 12, 16, 20])
    cfg.setdefault("yn_strengths", [0.2, 0.3, 0.4])
    cfg.setdefault("yn_kinds", ["concept", "random", "split"])
    L.log("extra3 phase", args.phase, "model", cfg["model"])
    L.set_wired_limit()
    lab = L.Lab(cfg)
    X2.install_end_hooks()
    phase_yn(cfg, lab)
    L.log("done; peak memory", L.peak_mem_gb(), "GB")


if __name__ == "__main__":
    main()
