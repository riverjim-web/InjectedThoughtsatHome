#!/usr/bin/env python3
"""
extra.py - two follow-up controls for the "steering just makes it say yes" objection.

  random2  Random vectors with no meaning, scaled to exactly the size of a real
           concept vector at each layer and strength, asked the usual detection
           question. The main run only had random vectors at layer 16, strength 0.3.
  yesbias  The usual setup, but the trial question is swapped for an unrelated
           factual yes/no question, with and without a concept injected. If
           injection makes gpt-oss say "yes" to anything, wrong "yes" answers to
           questions whose answer is "no" should rise with strength.

Same model, prompt format, sampling, token budget and injection start as the
main run (see lab.py). Usage: python extra.py random2|yesbias [--config cfg.json]
Writes results/inject_random2.jsonl and results/yesbias.jsonl (both resumable).
"""
import argparse
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab as L  # noqa: E402

# Unrelated questions. Eight have the answer "no", four "yes", so a pure pull
# toward "yes" and plain confusion can be told apart.
QUESTIONS = [
    ("Is Paris the capital of Germany?", "no"),
    ("Is 7 an even number?", "no"),
    ("Does the Moon orbit Mars?", "no"),
    ("Do spiders have six legs?", "no"),
    ("Is the Pacific the smallest ocean on Earth?", "no"),
    ("Is ice hotter than boiling water?", "no"),
    ("Is the Eiffel Tower in Rome?", "no"),
    ("Is a whale a fish?", "no"),
    ("Is Tokyo in Japan?", "yes"),
    ("Is 10 greater than 3?", "yes"),
    ("Does water freeze at 0 degrees Celsius?", "yes"),
    ("Is the Sun a star?", "yes"),
]
SUFFIX = " Answer yes or no."


def yesbias_messages(q):
    return [
        {"role": "user", "content": L.DETECT_SETUP},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": "Ok, let's get started.\n\nTrial 1: " + q + SUFFIX},
    ]


YN_RE = re.compile(r"\b(yes|no)\b", re.I)


def yes_no(final):
    """First yes/no word in the final answer, or None."""
    if not final:
        return None
    m = YN_RE.search(final.replace("*", ""))
    return m.group(1).lower() if m else None


def _done(path):
    ids = set()
    if path.exists():
        for line in open(path):
            try:
                ids.add(json.loads(line)["trial"])
            except Exception:
                pass
    return ids


def _run(cfg, lab, trials, out_name, make):
    """make(t) -> (ids, inject or None, extra dict). Generates, parses, appends."""
    out_path = L.RESULTS / out_name
    done_ids = _done(out_path)
    t0 = time.time()
    budget = cfg.get("extra_budget_min", 90) * 60
    done, n_tok, gen_secs = len(done_ids), 0, 0.0
    for t in trials:
        if t["trial"] in done_ids:
            continue
        if L.stop_requested() or time.time() - t0 > budget:
            L.log("stopping early:", "STOP file" if L.stop_requested() else "time budget")
            break
        try:
            ids, inject, extra = make(t)
            toks, secs, _ = lab.generate(ids, max_tokens=cfg["inject_max_tokens"], temp=cfg["temp"],
                                         seed=t["seed"], inject=inject)
            text = lab.decode(toks)
            an, fi = lab.parse(text)
            rec = dict(t)
            rec.update(extra)
            rec.update({"analysis": an, "final": fi, "text": text, "n_tokens": len(toks),
                        "secs": round(secs, 2), "hit_max": len(toks) >= cfg["inject_max_tokens"]})
            rec.update(L.quick_flags(t.get("concept"), an, fi))
            L.append_jsonl(out_path, rec)
            n_tok += len(toks)
            gen_secs += secs
        except Exception as e:
            L.append_jsonl(L.RESULTS / "errors.jsonl", {"phase": out_name, "trial": t["trial"],
                                                        "error": repr(e), "tb": traceback.format_exc()})
        done += 1
        if done % 5 == 0 or done == len(trials):
            L._progress(L.RESULTS / "progress.json", out_path.stem, done, len(trials), t0,
                        {"tok_per_s": round(n_tok / max(gen_secs, 1e-6), 1)})
    L._progress(L.RESULTS / "progress.json", out_path.stem, done, len(trials), t0,
                {"tok_per_s": round(n_tok / max(gen_secs, 1e-6), 1), "finished": True})
    L.log(f"{out_name}: {done}/{len(trials)} trials")


def _seeded(trials, seed, offset):
    random.Random(seed + offset).shuffle(trials)
    for i, t in enumerate(trials):
        t["trial"] = i
        t["seed"] = seed + offset + 7919 * (i + 1)
    return trials


def random_vector(vecs, norms, cfg, t):
    """Gaussian random direction, rescaled to the length of concept ci's vector at
    this layer and strength (the same recipe as the main run's random trials)."""
    ref = L.scaled(vecs[t["layer"]][t["ci"]], t["strength"], cfg, norms[t["layer"]])
    r = np.random.default_rng(t["seed"]).standard_normal(ref.shape[-1]).astype(np.float32)
    return r * (np.linalg.norm(ref) / np.linalg.norm(r))


def phase_random2(cfg, lab):
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    ids, start, _ = lab.prompt_ids(L.detection_messages(cfg), marker=L.DETECT_MARKER)
    skip = {(cfg["random_layer"], cfg["random_strength"])}       # already in the main run
    trials = [{"kind": "random", "layer": Lr, "strength": s_, "concept": None, "ci": j % len(concepts)}
              for Lr in cfg["layers"] for s_ in cfg["random2_strengths"] if (Lr, s_) not in skip
              for j in range(cfg["random2_per_cell"])]
    trials = _seeded(trials, cfg["seed"], 100003)
    L.write_json(L.RESULTS / "inject_random2_meta.json", {"n_trials": len(trials), "inject_start": start,
                                                          "n_prompt": len(ids), "cells": sorted({(t["layer"], t["strength"]) for t in trials})})
    _run(cfg, lab, trials, "inject_random2.jsonl",
         lambda t: (ids, {t["layer"]: (random_vector(vecs, norms, cfg, t), start)}, {}))


def phase_yesbias(cfg, lab):
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    prompts = {}
    for qi, (q, _) in enumerate(QUESTIONS):
        prompts[qi] = lab.prompt_ids(yesbias_messages(q), marker=L.DETECT_MARKER)[:2]
    trials = []
    for Lr in cfg["layers"]:
        for s_ in cfg["yesbias_strengths"]:
            for k, c in enumerate(concepts):
                qi = (k + 5 * cfg["layers"].index(Lr) + 3 * cfg["yesbias_strengths"].index(s_)) % len(QUESTIONS)
                trials.append({"kind": "inject", "layer": Lr, "strength": s_, "concept": c, "ci": k, "qi": qi})
    for j in range(cfg["yesbias_controls"]):
        trials.append({"kind": "control", "layer": None, "strength": 0, "concept": None, "ci": None,
                       "qi": j % len(QUESTIONS)})
    trials = _seeded(trials, cfg["seed"], 200003)
    L.write_json(L.RESULTS / "yesbias_meta.json", {"n_trials": len(trials), "questions": QUESTIONS,
                                                   "suffix": SUFFIX, "example_prompt": lab.render(yesbias_messages(QUESTIONS[0][0]))})

    def make(t):
        ids, start = prompts[t["qi"]]
        q, expect = QUESTIONS[t["qi"]]
        inject = None
        if t["kind"] == "inject":
            inject = {t["layer"]: (L.scaled(vecs[t["layer"]][t["ci"]], t["strength"], cfg, norms[t["layer"]]), start)}
        return ids, inject, {"question": q, "expected": expect}

    _run(cfg, lab, trials, "yesbias.jsonl", make)
    # add the parsed yes/no in a second pass (keeps _run generic)
    p = L.RESULTS / "yesbias.jsonl"
    rows = [json.loads(x) for x in open(p)]
    for r in rows:
        r["said"] = yes_no(r.get("final"))
        r["correct"] = r["said"] == r["expected"] if r["said"] else None
    with open(L.RESULTS / "yesbias_graded.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def summary():
    """Plain-text summary of both runs, printed into the job log."""
    out = []
    p = L.RESULTS / "inject_random2.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in open(p)]
        cells = {}
        for r in rows:
            cells.setdefault((r["layer"], r["strength"]), []).append(r)
        out.append(f"random2: {len(rows)} trials; first-pass 'affirm' flag by cell (hand-check before trusting):")
        for k in sorted(cells):
            g = cells[k]
            out.append(f"  L{k[0]} s{k[1]}: affirm {sum(r['affirm'] for r in g)}/{len(g)}, no final {sum(not r['final'] for r in g)}")
    p = L.RESULTS / "yesbias_graded.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in open(p)]
        out.append(f"yesbias: {len(rows)} trials")
        groups = {}
        for r in rows:
            key = "control" if r["kind"] == "control" else f"s{r['strength']}"
            groups.setdefault((key, r["expected"]), []).append(r)
        for (key, exp), g in sorted(groups.items()):
            said = [r["said"] for r in g]
            out.append(f"  {key:8s} answer={exp}: said yes {said.count('yes')}, no {said.count('no')}, "
                       f"neither {said.count(None)} (n={len(g)})")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["random2", "yesbias", "summary"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.phase == "summary":
        summary()
        return
    cfg = L.load_config(args.config)
    cfg.setdefault("random2_strengths", [0.3, 0.4, 0.6])
    cfg.setdefault("random2_per_cell", 36)
    cfg.setdefault("yesbias_strengths", [0.3, 0.6])
    cfg.setdefault("yesbias_controls", 72)
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    L.log("extra phase", args.phase, "model", cfg["model"])
    L.set_wired_limit()
    lab = L.Lab(cfg)
    if args.phase == "random2":
        phase_random2(cfg, lab)
    else:
        phase_yesbias(cfg, lab)
    L.log("done; peak memory", L.peak_mem_gb(), "GB")


if __name__ == "__main__":
    main()
