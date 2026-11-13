#!/usr/bin/env python3
"""
extra5.py - forced choice: does information about the injected concept reach the answer
when nothing is steering the answer?

The 36 concepts are listed as numbered options in a shuffled order, the final answer is
prefilled, and one forward pass gives the probability of each option number (1-36 are single
tokens). The vector is added from the start of the question to the end of the user's turn and
switched off before the assistant turn ("off"), or left on throughout ("on", an upper bound).

  detect   the usual preamble, then "Which of these concepts is the injected thought about?"
  neutral  no preamble, no mention of injection: "Pick one word from this list."
  word     like detect, but answered with the word (scored on the 36 words' first tokens,
           which are distinct), as a check that the numbered format isn't hiding anything
  matrix   (separate phase) "Is the injected thought about <word>?" for every injected concept x
           every word, vector off, with a twin question that only tests priming:
           'Does this sentence mention <word>? "The meeting was moved to Thursday afternoon."' 

Vectors: concept (off and on), split and random (off), layers 12 and 16, strengths 0.2-0.4,
two shuffles per concept. Each shuffle gets its own no-injection control.
Usage: python extra5.py fc|summary [--config cfg.json]   ->  results/forced_choice.jsonl
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab as L  # noqa: E402
import extra2 as X2  # noqa: E402

NEUTRAL_MARK = "Here is a list of words."


def option_block(words):
    return "\n".join(f"{i + 1}. {w}" for i, w in enumerate(words))


def detect_msgs(words, answer="number"):
    tail = "Reply with the number only." if answer == "number" else "Reply with the word only, in lowercase."
    q = "Which of these concepts is the injected thought about? " + tail + "\n\n" + option_block(words)
    return [{"role": "user", "content": L.DETECT_SETUP}, {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": "Ok, let's get started.\n\nTrial 1: " + q}]


def neutral_msgs(words, answer="number"):
    tail = "Reply with its number only." if answer == "number" else "Reply with the word only, in lowercase."
    return [{"role": "user", "content": NEUTRAL_MARK + "\n\n" + option_block(words) +
             "\n\nPick one word from the list. " + tail}]


def match_msgs(word):
    return [{"role": "user", "content": L.DETECT_SETUP}, {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": f"Ok, let's get started.\n\nTrial 1: Is the injected thought about {word}? Answer yes or no."}]


SENTENCE = "The meeting was moved to Thursday afternoon."


def matrix_msgs(word, frame):
    if frame == "introspect":
        q = f"Is the injected thought about {word}? Answer yes or no."
    else:   # same form and window, no introspective content; the true answer is always no
        q = f'Does this sentence mention {word}? "{SENTENCE}" Answer yes or no.'
    return [{"role": "user", "content": L.DETECT_SETUP}, {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": "Ok, let's get started.\n\nTrial 1: " + q}]


def phase_matrix(cfg, lab):
    """Every injected concept x every asked word, vector switched off before the answer.
    Diagonal (asked word = injected concept) vs off-diagonal, for the introspective question and
    for a twin question that only tests priming."""
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    Y, N, _ = X2.yes_no_sets(lab)
    P = {}
    for fr in ("introspect", "sentence"):
        for w in range(len(concepts)):
            s = lab.render(matrix_msgs(concepts[w], fr)) + X2.FINAL_PREFILL
            P[(fr, w)] = X2.positions(lab, s, L.DETECT_MARKER)
    out = L.RESULTS / "match_matrix.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    jobs = [{"frame": fr, "kind": "control", "layer": None, "strength": 0, "ci": None, "ask": w}
            for fr in ("introspect", "sentence") for w in range(len(concepts))]
    for Lr in cfg["mx_layers"]:
        for s_ in cfg["mx_strengths"]:
            for ci in range(len(concepts)):
                for fr in ("introspect", "sentence"):
                    for w in range(len(concepts)):
                        jobs.append({"frame": fr, "kind": "concept", "layer": Lr, "strength": s_, "ci": ci, "ask": w})
    for j in jobs:
        j["id"] = "|".join(str(j.get(k)) for k in ("frame", "kind", "layer", "strength", "ci", "ask"))
    L.write_json(L.RESULTS / "match_matrix_meta.json", {"n_jobs": len(jobs), "concepts": concepts, "sentence": SENTENCE,
                 "example_introspect": lab.render(matrix_msgs(concepts[1], "introspect")) + X2.FINAL_PREFILL,
                 "example_sentence": lab.render(matrix_msgs(concepts[1], "sentence")) + X2.FINAL_PREFILL})
    vcache = {}
    t0 = time.time()
    for n_done, j in enumerate(jobs):
        if j["id"] in done:
            continue
        if L.stop_requested():
            break
        try:
            ids, start, end_user = P[(j["frame"], j["ask"])]
            inject = None
            if j["kind"] != "control":
                vk = (j["layer"], j["strength"], j["ci"])
                if vk not in vcache:
                    vcache[vk] = X2.make_vector("concept", j["layer"], j["strength"], j["ci"], 0, cfg, vecs, norms, None)
                inject = {j["layer"]: (vcache[vk], start)}
            p = X2.first_token_probs(lab, ids, inject, end_user if inject else None)
            rec = dict(j)
            rec.update({"p_yes": float(p[Y].sum()), "p_no": float(p[N].sum()), "diag": j["ci"] == j["ask"]})
            L.append_jsonl(out, rec)
        except Exception as e:
            L.append_jsonl(L.RESULTS / "errors.jsonl", {"phase": "matrix", "id": j["id"], "error": repr(e), "tb": traceback.format_exc()})
        if n_done % 50 == 0:
            L._progress(L.RESULTS / "progress.json", "match_matrix", n_done, len(jobs), t0)
    L._progress(L.RESULTS / "progress.json", "match_matrix", len(jobs), len(jobs), t0, {"finished": True})


def phase_fc(cfg, lab):
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    B, _ = X2.load_baseline()
    Y, N, _ = X2.yes_no_sets(lab)
    num_ids = np.array([lab.encode(str(n))[0] for n in range(1, len(concepts) + 1)])
    word_ids = np.array([lab.encode(c)[0] for c in concepts])
    if not cfg.get("fc_test_model"):        # the tiny test model's byte-level tokenizer can't pass these
        assert all(len(lab.encode(str(n))) == 1 for n in range(1, len(concepts) + 1))
        assert len(set(word_ids.tolist())) == len(concepts), "concept first tokens collide"
    n_sh = cfg["fc_shuffles"]
    perms = {}
    for ci in range(len(concepts)):
        for sh in range(n_sh):
            p = list(range(len(concepts)))
            random.Random(cfg["seed"] * 7 + 1009 * ci + 31 * sh).shuffle(p)
            perms[(ci, sh)] = p                      # perms[...][option_index] = concept index

    cache = {}

    def prompt(fr, ci, sh):
        key = (fr, ci, sh)
        if key not in cache:
            words = [concepts[j] for j in perms[(ci, sh)]] if fr != "match" else None
            if fr == "detect":
                msgs, mark = detect_msgs(words), L.DETECT_MARKER
            elif fr == "word":
                msgs, mark = detect_msgs(words, "word"), L.DETECT_MARKER
            elif fr == "neutral":
                msgs, mark = neutral_msgs(words), NEUTRAL_MARK
            elif fr == "nword":
                msgs, mark = neutral_msgs(words, "word"), NEUTRAL_MARK
            else:                                    # ("match", word index, 0)
                msgs, mark = match_msgs(concepts[ci]), L.DETECT_MARKER
            s = lab.render(msgs) + X2.FINAL_PREFILL
            cache[key] = X2.positions(lab, s, mark)
        return cache[key]

    jobs = []
    if cfg.get("fc_round") == 2:
        # round 2: the neutral framing answered with the word (does the detection framing add anything?),
        # and split/random vectors in the word format (the paired concept should sit at chance)
        for ci in range(len(concepts)):
            for sh in range(n_sh):
                jobs.append({"frame": "nword", "kind": "control", "timing": None, "layer": None, "strength": 0, "ci": ci, "sh": sh})
        for Lr in cfg["fc_layers"]:
            for s_ in cfg["fc_strengths"]:
                for ci in range(len(concepts)):
                    for sh in range(n_sh):
                        base = {"layer": Lr, "strength": s_, "ci": ci, "sh": sh}
                        jobs += [dict(base, frame="nword", kind="concept", timing="off"),
                                 dict(base, frame="nword", kind="concept", timing="on"),
                                 dict(base, frame="word", kind="split", timing="off"),
                                 dict(base, frame="word", kind="random", timing="off")]
    for fr in (() if cfg.get("fc_round") == 2 else ("detect", "word", "neutral")):
        for ci in range(len(concepts)):
            for sh in range(n_sh):
                jobs.append({"frame": fr, "kind": "control", "timing": None, "layer": None, "strength": 0, "ci": ci, "sh": sh})
    for Lr in ([] if cfg.get("fc_round") == 2 else cfg["fc_layers"]):
        for s_ in cfg["fc_strengths"]:
            for ci in range(len(concepts)):
                for sh in range(n_sh):
                    base = {"layer": Lr, "strength": s_, "ci": ci, "sh": sh}
                    jobs += [dict(base, frame="detect", kind="concept", timing="off"),
                             dict(base, frame="detect", kind="concept", timing="on"),
                             dict(base, frame="detect", kind="split", timing="off"),
                             dict(base, frame="detect", kind="random", timing="off"),
                             dict(base, frame="neutral", kind="concept", timing="off"),
                             dict(base, frame="neutral", kind="concept", timing="on"),
                             dict(base, frame="word", kind="concept", timing="off"),
                             dict(base, frame="word", kind="concept", timing="on")]
    for j in jobs:
        j["id"] = "|".join(str(j.get(k)) for k in ("frame", "kind", "timing", "layer", "strength", "ci", "sh", "ask"))
    out = L.RESULTS / "forced_choice.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    meta_name = "forced_choice_meta2.json" if cfg.get("fc_round") == 2 else "forced_choice_meta.json"
    L.write_json(L.RESULTS / meta_name, {
        "n_jobs": len(jobs), "concepts": concepts, "perms": {f"{k[0]}|{k[1]}": v for k, v in perms.items()},
        "example_detect": lab.render(detect_msgs([concepts[j] for j in perms[(0, 0)]])) + X2.FINAL_PREFILL,
        "example_neutral": lab.render(neutral_msgs([concepts[j] for j in perms[(0, 0)]])) + X2.FINAL_PREFILL,
        "example_match": lab.render(match_msgs(concepts[0])) + X2.FINAL_PREFILL,
        "example_nword": lab.render(neutral_msgs([concepts[j] for j in perms[(0, 0)]], "word")) + X2.FINAL_PREFILL})
    vcache = {}
    t0 = time.time()
    for n_done, j in enumerate(jobs):
        if j["id"] in done:
            continue
        if L.stop_requested():
            break
        try:
            if j["frame"] == "match":
                ids, start, end_user = prompt("match", j["ask"], 0)
            else:
                ids, start, end_user = prompt(j["frame"], j["ci"], j["sh"])
            inject, end = None, None
            if j["kind"] != "control":
                vk = (j["kind"], j["layer"], j["strength"], j["ci"])
                if vk not in vcache:
                    seed = cfg["seed"] + 104729 * (j["ci"] + 1) + j["layer"]
                    vcache[vk] = X2.make_vector(j["kind"], j["layer"], j["strength"], j["ci"], seed, cfg, vecs, norms, B)
                inject = {j["layer"]: (vcache[vk], start)}
                end = end_user if j["timing"] == "off" else None
            p = X2.first_token_probs(lab, ids, inject, end)
            rec = dict(j)
            if j["frame"] == "match":
                rec.update({"p_yes": float(p[Y].sum()), "p_no": float(p[N].sum()), "match": j["ask"] == j["ci"]})
            else:
                perm = perms[(j["ci"], j["sh"])]
                if j["frame"] in ("word", "nword"):
                    pw = p[word_ids]                       # indexed by concept
                    opt = [float(pw[c]) for c in perm]      # reorder to option order
                else:
                    opt = [float(x) for x in p[num_ids]]    # option order
                rec.update({"opt_p": opt, "mass": float(sum(opt)), "target_opt": perm.index(j["ci"])})
            L.append_jsonl(out, rec)
        except Exception as e:
            L.append_jsonl(L.RESULTS / "errors.jsonl", {"phase": "fc", "id": j["id"], "error": repr(e), "tb": traceback.format_exc()})
        if n_done % 25 == 0:
            L._progress(L.RESULTS / "progress.json", "forced_choice", n_done, len(jobs), t0)
    L._progress(L.RESULTS / "progress.json", "forced_choice", len(jobs), len(jobs), t0, {"finished": True})


def summary():
    p = L.RESULTS / "forced_choice.jsonl"
    if not p.exists():
        print("no forced_choice.jsonl")
        return
    rows = [json.loads(x) for x in open(p)]
    ctl = {(r["frame"], r["ci"], r["sh"]): r for r in rows if r["kind"] == "control" and r["frame"] != "match"}
    groups = {}
    for r in rows:
        if r["frame"] == "match" or r["kind"] == "control":
            continue
        groups.setdefault((r["frame"], r["kind"], r["timing"], r["strength"]), []).append(r)
    print("frame   kind    timing s  : top-1 acc | mean rank | mean shift in log P(target) vs control | n")
    for k in sorted(groups, key=str):
        g = groups[k]
        acc, rank, shift = [], [], []
        for r in g:
            q = np.array(r["opt_p"]); q = q / q.sum()
            c = np.array(ctl[(r["frame"], r["ci"], r["sh"])]["opt_p"]); c = c / c.sum()
            t = r["target_opt"]
            acc.append(int(np.argmax(q) == t)); rank.append(int((q > q[t]).sum()) + 1)
            shift.append(float(np.log(q[t] + 1e-12) - np.log(c[t] + 1e-12)))
        print(f"  {k[0]:7s} {k[1]:7s} {str(k[2]):4s} {k[3]}: {np.mean(acc):.3f} | {np.mean(rank):5.1f} | {np.mean(shift):+.2f} | {len(g)}")
    mp = L.RESULTS / "match_matrix.jsonl"
    if mp.exists():
        mrows = [json.loads(x) for x in open(mp)]
        lo = lambda r: np.log(r["p_yes"] + 1e-9) - np.log(r["p_no"] + 1e-9)
        c = {(r["frame"], r["ask"]): lo(r) for r in mrows if r["kind"] == "control"}
        for fr in ("introspect", "sentence"):
            for Lr in sorted({r["layer"] for r in mrows if r["kind"] != "control"}):
                for s_ in sorted({r["strength"] for r in mrows if r["kind"] != "control"}):
                    g = [r for r in mrows if r["frame"] == fr and r["kind"] != "control" and r["layer"] == Lr and r["strength"] == s_]
                    d = [lo(r) - c[(fr, r["ask"])] for r in g if r["diag"]]
                    o = [lo(r) - c[(fr, r["ask"])] for r in g if not r["diag"]]
                    if d:
                        print(f"  matrix {fr:10s} L{Lr} s{s_}: diagonal shift {np.mean(d):+.2f}, off-diagonal {np.mean(o):+.2f}, "
                              f"difference {np.mean(d) - np.mean(o):+.2f} (n={len(d)}/{len(o)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fc", "matrix", "summary"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.phase == "summary":
        summary()
        return
    cfg = L.load_config(args.config)
    cfg.setdefault("fc_layers", [12, 16])
    cfg.setdefault("fc_strengths", [0.2, 0.3, 0.4])
    cfg.setdefault("fc_shuffles", 2)
    cfg.setdefault("mx_layers", [12, 16])
    cfg.setdefault("mx_strengths", [0.3])
    L.log("extra5 phase", args.phase, "model", cfg["model"])
    L.set_wired_limit()
    lab = L.Lab(cfg)
    X2.install_end_hooks()
    (phase_fc if args.phase == "fc" else phase_matrix)(cfg, lab)
    L.log("done; peak memory", L.peak_mem_gb(), "GB")


if __name__ == "__main__":
    main()
