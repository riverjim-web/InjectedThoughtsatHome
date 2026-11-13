#!/usr/bin/env python3
"""
extra2.py - the adversarial-review runs.

  baseline  Recompute per-word activations for the 80 baseline words (the main run
            only saved their average). Needed for the two structured null vectors.
  logits    One forward pass per trial, no sampling: the probability that gpt-oss's
            final answer starts with "Yes" vs "No", read at the first answer token
            (the final channel is prefilled, so there is no reasoning to read first).
            Two timings: "on" (vector added through the read position, as in the main
            run) and "off" (vector added only while the model reads the user's turn,
            switched off before the assistant turn begins, so nothing it says or reads
            of its own is steered). Vector types: concept, random (Gaussian),
            split (difference of two random halves of the baseline words: shaped like
            real activations, meaningless) and baseword (a mundane baseline word's own
            vector, e.g. "kettles"). Also the same readout on three unrelated
            factual questions whose answer is "no", as a yes-bias control.
  latent    Sampled detection trials with the concept vector switched off before the
            assistant turn (the remove-before-answer design), graded like the main run.
  null2     Sampled detection trials, vector on throughout like the main run, with the
            two structured null vectors (split and baseword).

Usage: python extra2.py baseline|logits|latent|null2|summary [--config cfg.json]
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
import lab as L  # noqa: E402
import extra as E  # noqa: E402

# ---------------------------------------------------------------- hooks with an end position
END = {}   # layer -> first position NOT injected (absent = inject to the end)


def _call(self, x, *args, **kwargs):
    out = L._BASE_CLASS[type(self)].__call__(self, x, *args, **kwargs)
    i = L._LAYER_INDEX.get(id(self))
    if i is None:
        return out
    is_tuple = isinstance(out, tuple)
    h = out[0] if is_tuple else out
    n = h.shape[1]
    p0 = L.STATE.pos.get(i, 0)
    L.STATE.pos[i] = p0 + n
    inj = L.STATE.inject.get(i)
    if inj is not None:
        vec, start = inj
        end = END.get(i, 1 << 30)
        lo, hi = max(start, p0), min(end, p0 + n)
        if hi > lo:
            v = vec.astype(h.dtype)
            if lo == p0 and hi == p0 + n:
                h = h + v
            else:
                ar = mx.arange(p0, p0 + n)
                m = ((ar >= start) & (ar < end)).astype(h.dtype)
                h = h + m[None, :, None] * v
    mode = L.STATE.capture.get(i)
    if mode is not None:
        L.STATE.captured.setdefault(i, []).append(h[:, -1:, :] if mode == "last" else h)
    if is_tuple:
        return (h,) + tuple(out[1:])
    return h


def install_end_hooks():
    for cls in list(L._BASE_CLASS):
        cls.__call__ = _call


# ---------------------------------------------------------------- prompts
FINAL_PREFILL = "<|channel|>final<|message|>"
YB_QUESTIONS = ["Is Paris the capital of Germany?", "Does the Moon orbit Mars?", "Is the Eiffel Tower in Rome?"]


def positions(lab, s, marker):
    """Token ids for string s, the first token of `marker`, and the first token of
    the assistant header (where 'off' injection stops)."""
    full = lab.encode(s)

    def tok_index(idx):
        pre = lab.encode(s[:idx])
        k = 0
        while k < len(pre) and k < len(full) and pre[k] == full[k]:
            k += 1
        return k
    start = tok_index(s.rfind(marker))
    end_user = tok_index(s.rfind("<|start|>assistant"))
    return full, start, end_user


def detect_prompt(lab, cfg, prefill=True):
    s = lab.render(L.detection_messages(cfg)) + (FINAL_PREFILL if prefill else "")
    return positions(lab, s, L.DETECT_MARKER)


def question_prompt(lab, q):
    s = lab.render(E.yesbias_messages(q)) + FINAL_PREFILL
    return positions(lab, s, L.DETECT_MARKER)


# ---------------------------------------------------------------- vectors
def load_baseline():
    z = np.load(L.RESULTS / "baseline_acts.npz", allow_pickle=False)
    return {int(k[2:]): z[k] for k in z.files if k.startswith("B_")}, [str(w) for w in z["words"]]


def make_vector(kind, layer, strength, ci, seed, cfg, vecs, norms, B):
    if kind == "concept":
        base = vecs[layer][ci]
    elif kind == "random":
        base = np.random.default_rng(seed).standard_normal(vecs[layer].shape[-1]).astype(np.float32)
    elif kind == "split":
        A = B[layer]
        idx = np.random.default_rng(seed).permutation(len(A))
        h = len(A) // 2
        base = A[idx[:h]].mean(axis=0) - A[idx[h:]].mean(axis=0)
    elif kind == "baseword":
        A = B[layer]
        j = ci % len(A)
        base = A[j] - np.delete(A, j, axis=0).mean(axis=0)
    else:
        raise ValueError(kind)
    return L.scaled(np.asarray(base, dtype=np.float32), strength, cfg, norms[layer]).astype(np.float32)


# ---------------------------------------------------------------- phases
def phase_baseline(cfg, lab):
    layers = sorted(set(cfg["x2_layers"]))
    A, texts = L.response_mean_acts(lab, cfg["baseline"], layers, max_tokens=cfg.get("resp_tokens", 96))
    np.savez_compressed(L.RESULTS / "baseline_acts.npz", words=np.array(cfg["baseline"]),
                        **{f"B_{Lr}": A[Lr].astype(np.float32) for Lr in layers})
    # sanity: greedy decoding is deterministic, so the average of these should reproduce the
    # baseline mean saved with the main run's concept vectors
    _, means, _ = L.load_vectors(L.RESULTS / "vectors.npz")
    for Lr in layers:
        a, b = A[Lr].mean(axis=0), means[Lr]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        L.log(f"baseline acts L{Lr}: {A[Lr].shape}, cosine with saved baseline mean {cos:.5f}, "
              f"relative difference {float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8)):.5f}")


def yes_no_sets(lab):
    yes, no, i_ = [], [], []
    vocab = lab.hf.get_vocab()
    for tok, tid in vocab.items():
        s = lab.hf.decode([tid]).strip().strip("*").strip().lower()
        if s == "yes":
            yes.append(tid)
        elif s == "no":
            no.append(tid)
        elif s == "i":
            i_.append(tid)
    return np.array(yes, dtype=np.int64), np.array(no, dtype=np.int64), np.array(i_, dtype=np.int64)


def first_token_probs(lab, ids, inject=None, end=None):
    L.STATE.clear()
    END.clear()
    if inject:
        L.STATE.inject = {Lr: (mx.array(v), int(st)) for Lr, (v, st) in inject.items()}
        if end is not None:
            END.update({Lr: int(end) for Lr in inject})
    logits = lab.model(mx.array(ids)[None])[0, -1].astype(mx.float32)
    p = mx.softmax(logits)
    mx.eval(p)
    L.STATE.clear()
    END.clear()
    return np.array(p)


def phase_logits(cfg, lab):
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    B, words = load_baseline()
    Y, N, I_ = yes_no_sets(lab)
    L.log(f"yes tokens {len(Y)}, no tokens {len(N)}, I tokens {len(I_)}")
    prompts = {"detect": detect_prompt(lab, cfg)}
    for qi, q in enumerate(YB_QUESTIONS):
        prompts[f"q{qi}"] = question_prompt(lab, q)
    L.write_json(L.RESULTS / "logits_meta.json", {
        "prompts": {k: {"n": len(v[0]), "start": v[1], "end_user": v[2], "text_tail": lab.decode(v[0][-60:])} for k, v in prompts.items()},
        "yes_tokens": [lab.hf.decode([int(t)]) for t in Y], "no_tokens": [lab.hf.decode([int(t)]) for t in N]})
    out = L.RESULTS / "logits.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    jobs = []
    for pk in prompts:
        jobs.append({"id": f"{pk}|control", "prompt": pk, "kind": "control", "timing": None, "layer": None, "strength": 0, "ci": None})
        kinds = cfg["x2_kinds"] if pk == "detect" else ["concept"]
        strengths = cfg["x2_strengths"] if pk == "detect" else cfg["x2_q_strengths"]
        for kind in kinds:
            for Lr in cfg["x2_layers"]:
                for s_ in strengths:
                    for timing in ("on", "off"):
                        for ci in range(len(concepts)):
                            jobs.append({"id": f"{pk}|{kind}|{Lr}|{s_}|{timing}|{ci}", "prompt": pk, "kind": kind,
                                         "timing": timing, "layer": Lr, "strength": s_, "ci": ci})
    t0 = time.time()
    for n_done, j in enumerate(jobs):
        if j["id"] in done:
            continue
        if L.stop_requested():
            break
        ids, start, end_user = prompts[j["prompt"]]
        try:
            inject, end = None, None
            if j["kind"] != "control":
                seed = cfg["seed"] + 104729 * (j["ci"] + 1) + j["layer"]
                v = make_vector(j["kind"], j["layer"], j["strength"], j["ci"], seed, cfg, vecs, norms, B)
                inject = {j["layer"]: (v, start)}
                end = end_user if j["timing"] == "off" else None
            p = first_token_probs(lab, ids, inject, end)
            top = np.argsort(-p)[:5]
            rec = dict(j)
            rec.update({"p_yes": float(p[Y].sum()), "p_no": float(p[N].sum()), "p_I": float(p[I_].sum()),
                        "concept": concepts[j["ci"]] if j["kind"] == "concept" else (words[j["ci"] % len(words)] if j["kind"] == "baseword" else None),
                        "top": [[lab.hf.decode([int(t)]), round(float(p[t]), 4)] for t in top]})
            L.append_jsonl(out, rec)
        except Exception as e:
            L.append_jsonl(L.RESULTS / "errors.jsonl", {"phase": "logits", "id": j["id"], "error": repr(e), "tb": traceback.format_exc()})
        if n_done % 50 == 0:
            L._progress(L.RESULTS / "progress.json", "logits", n_done, len(jobs), t0)
    L._progress(L.RESULTS / "progress.json", "logits", len(jobs), len(jobs), t0, {"finished": True})


def _sampled(cfg, lab, trials, out_name, timing):
    vecs, _, concepts = L.load_vectors(L.RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in L.vector_meta()["resid_norm"].items()}
    B, words = load_baseline()
    ids, start, end_user = detect_prompt(lab, cfg, prefill=False)
    old_ids, old_start, _ = lab.prompt_ids(L.detection_messages(cfg), marker=L.DETECT_MARKER)
    assert ids == old_ids and start == old_start, "prompt differs from the main run"
    out_path = L.RESULTS / out_name
    done_ids = E._done(out_path)
    t0 = time.time()
    for n_done, t in enumerate(trials):
        if t["trial"] in done_ids:
            continue
        if L.stop_requested():
            break
        try:
            v = make_vector(t["vkind"], t["layer"], t["strength"], t["ci"], t["seed"], cfg, vecs, norms, B)
            END.clear()
            if timing == "off":
                END[t["layer"]] = end_user
            toks, secs, _ = lab.generate(ids, max_tokens=cfg["inject_max_tokens"], temp=cfg["temp"],
                                         seed=t["seed"], inject={t["layer"]: (v, start)})
            END.clear()
            text = lab.decode(toks)
            an, fi = lab.parse(text)
            rec = dict(t)
            rec["concept"] = concepts[t["ci"]] if t["vkind"] == "concept" else None
            rec["vector_word"] = words[t["ci"] % len(words)] if t["vkind"] == "baseword" else None
            rec.update({"timing": timing, "analysis": an, "final": fi, "text": text, "n_tokens": len(toks),
                        "secs": round(secs, 2), "hit_max": len(toks) >= cfg["inject_max_tokens"]})
            rec.update(L.quick_flags(rec["concept"], an, fi))
            L.append_jsonl(out_path, rec)
        except Exception as e:
            L.append_jsonl(L.RESULTS / "errors.jsonl", {"phase": out_name, "trial": t["trial"], "error": repr(e), "tb": traceback.format_exc()})
        if n_done % 5 == 0:
            L._progress(L.RESULTS / "progress.json", Path(out_name).stem, n_done, len(trials), t0)
    L._progress(L.RESULTS / "progress.json", Path(out_name).stem, len(trials), len(trials), t0, {"finished": True})


def _grid(cfg, kinds, offset):
    trials = [{"kind": "inject" if k == "concept" else k, "vkind": k, "layer": Lr, "strength": s_, "ci": ci}
              for k in kinds for Lr in cfg["x2_layers"] for s_ in cfg["x2_strengths"] for ci in range(len(cfg["concepts"]))]
    return E._seeded(trials, cfg["seed"], offset)


def phase_latent(cfg, lab):
    _sampled(cfg, lab, _grid(cfg, ["concept"], 300007), "latent.jsonl", "off")


def phase_null2(cfg, lab):
    _sampled(cfg, lab, _grid(cfg, ["split", "baseword"], 400009), "null2.jsonl", "on")


def summary():
    out = []
    p = L.RESULTS / "logits.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in open(p)]
        out.append(f"logits: {len(rows)} readouts")
        groups = {}
        for r in rows:
            key = (r["prompt"], r["kind"], r["timing"], r["strength"])
            groups.setdefault(key, []).append(r)
        for k in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]), k[3])):
            g = groups[k]
            py = np.array([r["p_yes"] for r in g]); pn = np.array([r["p_no"] for r in g])
            out.append(f"  {k[0]:6s} {k[1]:8s} {str(k[2]):4s} s{k[3]}: mean P(yes) {py.mean():.4f}  median {np.median(py):.4f}  "
                       f"mean P(no) {pn.mean():.4f}  n={len(g)}")
    for name in ("latent.jsonl", "null2.jsonl"):
        p = L.RESULTS / name
        if p.exists():
            rows = [json.loads(x) for x in open(p)]
            g = {}
            for r in rows:
                g.setdefault((r["vkind"], r["strength"]), []).append(r)
            out.append(f"{name}: {len(rows)} trials (first-pass 'affirm' flag; hand-check before trusting)")
            for k in sorted(g):
                out.append(f"  {k[0]:8s} s{k[1]}: affirm {sum(r['affirm'] for r in g[k])}/{len(g[k])}, no final {sum(not r['final'] for r in g[k])}")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["baseline", "logits", "latent", "null2", "summary"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.phase == "summary":
        summary()
        return
    cfg = L.load_config(args.config)
    cfg.setdefault("x2_layers", [12, 16])
    cfg.setdefault("x2_strengths", [0.3, 0.4, 0.6])
    cfg.setdefault("x2_q_strengths", [0.3, 0.6])
    cfg.setdefault("x2_kinds", ["concept", "random", "split", "baseword"])
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    L.log("extra2 phase", args.phase, "model", cfg["model"])
    L.set_wired_limit()
    lab = L.Lab(cfg)
    install_end_hooks()
    {"baseline": phase_baseline, "logits": phase_logits, "latent": phase_latent, "null2": phase_null2}[args.phase](cfg, lab)
    L.log("done; peak memory", L.peak_mem_gb(), "GB")


if __name__ == "__main__":
    main()
