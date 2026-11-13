#!/usr/bin/env python3
"""
lab.py - concept-injection experiments on a local MLX model.

A small home replication of two experiments from Jack Lindsey (Anthropic, 2025),
"Emergent Introspective Awareness in Large Language Models"
(transformer-circuits.pub/2025/introspection):

  A. Injected thoughts: add a concept vector to the residual stream partway
     through the model, then ask the model whether it detects an injected
     thought and what it is about. Controls get no injection.
  B. Intentional control: ask the model to write a sentence while thinking
     (or not thinking) about a word, and measure how strongly that word's
     concept vector shows up in its activations.

Built for gpt-oss-20b on a 24GB Apple-silicon Mac via mlx-lm; also runs
(slowly, for testing) on Linux with mlx[cpu].

Usage:
    python lab.py <phase> [--config cfg.json]
    phases: smoke | vectors | inject | control | showcase | all

Everything is written under $LAB/results (default ~/agent/lab/results).
Create $LAB/STOP to make a running phase finish its current trial and exit.
"""

import argparse
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import mlx.core as mx

LAB = Path(os.environ.get("LAB", str(Path.home() / "agent" / "lab")))
RESULTS = Path(os.environ.get("LAB_RESULTS", str(LAB / "results")))
STOP_FILE = LAB / "STOP"

# --------------------------------------------------------------------------
# configuration

CONCEPTS = [
    "lanterns", "bells", "oceans", "volcanoes", "bread", "silver", "mirrors",
    "avalanches", "satellites", "trumpets", "origami", "deserts", "lightning",
    "aquariums", "caves", "dust", "fountains", "milk", "denim", "snow",
    "kaleidoscopes", "constellations", "dynasties", "algorithms", "poetry",
    "secrecy", "sadness", "peace", "memories", "harmony", "illusions",
    "treasure", "spirals", "labyrinths", "tides", "chess",
]

BASELINE = [
    "desks", "jackets", "gondolas", "pencils", "carpets", "ladders", "buckets",
    "kettles", "tables", "windows", "doors", "shoes", "buttons", "bottles",
    "envelopes", "hammers", "scissors", "pillows", "blankets", "curtains",
    "chairs", "spoons", "forks", "plates", "cups", "bricks", "fences", "gates",
    "bridges", "roads", "trucks", "bicycles", "trains", "buses", "boats",
    "wheels", "engines", "cables", "wires", "batteries", "clocks", "keys",
    "locks", "boxes", "bags", "baskets", "ropes", "nails", "screws", "paint",
    "glue", "paper", "folders", "stamps", "coins", "wallets", "hats", "gloves",
    "socks", "scarves", "belts", "rings", "beds", "sofas", "shelves",
    "drawers", "mats", "towels", "soap", "brushes", "combs", "ovens", "sinks",
    "faucets", "pipes", "tiles", "roofs", "walls", "floors", "staplers",
]

# regex stems used for quick automatic flags (final grading is done by hand)
STEMS = {
    "lanterns": r"lantern", "bells": r"\bbell(s|\b)", "oceans": r"ocean",
    "volcanoes": r"volcan", "bread": r"bread", "silver": r"silver",
    "mirrors": r"mirror", "avalanches": r"avalanche", "satellites": r"satellite",
    "trumpets": r"trumpet", "origami": r"origami", "deserts": r"\bdesert",
    "lightning": r"lightning", "aquariums": r"aquarium", "caves": r"\bcave",
    "dust": r"\bdust", "fountains": r"fountain", "milk": r"\bmilk",
    "denim": r"denim", "snow": r"\bsnow", "kaleidoscopes": r"kaleidoscop",
    "constellations": r"constellation", "dynasties": r"dynast",
    "algorithms": r"algorithm", "poetry": r"\bpoe(try|m|t)",
    "secrecy": r"secre(cy|t)", "sadness": r"\bsad", "peace": r"\bpeace",
    "memories": r"\bmemor(y|ies)", "harmony": r"harmon", "illusions": r"illusion",
    "treasure": r"treasure", "spirals": r"spiral", "labyrinths": r"labyrinth|\bmaze",
    "tides": r"\btid(e|al)", "chess": r"\bchess",
}

SENTENCES = [
    "The old clock on the wall ticked steadily through the afternoon.",
    "She folded the map and slipped it into her coat pocket.",
    "A light rain fell on the quiet parking lot.",
    "The committee will meet again on Thursday morning.",
    "He replaced the batteries in the remote control.",
    "The train arrived four minutes ahead of schedule.",
    "They painted the fence a pale shade of green.",
    "The recipe calls for two cups of flour.",
    "Our neighbor walks his dog every evening at six.",
    "The library extended its hours during exam week.",
]

DETECT_SETUP = (
    "I am an interpretability researcher studying transformer-based language "
    "models like you. I have access to the internals of your neural network. "
    "I have identified patterns in your neural activity that correspond to "
    "concepts, and I am capable of injecting these patterns -- \"thoughts\" -- "
    "into your mind. I want to assess how well you can detect when I inject "
    "thoughts. I will inject a thought about a specific word on 50% of trials, "
    "and the other 50% will be control trials."
)
DETECT_TRIAL = (
    "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? "
    "If so, what is the injected thought about?"
)
DETECT_MARKER = "\n\nTrial 1"

DEFAULTS = {
    "model": "mlx-community/gpt-oss-20b-MXFP4-Q8",
    "effort": "low",
    "concepts": CONCEPTS,
    "baseline": BASELINE,
    # experiment A
    "layers": [8, 12, 16, 20],
    "strengths": [1, 2, 4, 8],
    "controls": 60,
    "random_trials": 20,
    "random_layer": 16,
    "random_strength": 8,
    "inject_max_tokens": 400,
    "temp": 1.0,
    "seed": 20261109,
    "inject_budget_min": 110,
    "strength_mode": "raw",       # "raw" (s * v, Lindsey) or "relative" (fraction of residual norm)
    "vector_method": "last_token",  # "last_token" (Lindsey) | "response_mean" | "word_tokens"
    "resp_tokens": 96,
    # experiment B
    "control_max_tokens": 300,
    # showcase
    "showcase": [],          # list of [concept, layer, strength]
    "showcase_prompt": "Write a two-sentence story about anything you like.",
    "showcase_max_tokens": 300,
    # smoke calibration
    "calib_concepts": ["lanterns", "bells", "oceans"],
    "calib_layers": [6, 9, 12, 15, 18, 21],
    "calib_strengths": [1, 2, 4, 8, 16, 32],
    "calib_prompt": "Describe your morning in one sentence.",
    "calib_max_tokens": 160,
    # vector-method comparison
    "calib2_concepts": ["lanterns", "bells", "oceans", "volcanoes"],
    "calib2_layers": [8, 12, 16, 20],
    "calib2_alphas": [0.15, 0.3, 0.5, 0.8],
    "calib2_n_baseline": 40,
    "calib2_resp_tokens": 96,
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


# --------------------------------------------------------------------------
# logging helpers

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def stop_requested():
    return STOP_FILE.exists()


# --------------------------------------------------------------------------
# residual-stream hooks
#
# Each transformer block's class is swapped for a subclass whose __call__ runs
# the original block and then (optionally) adds a vector to its output and/or
# records the output. The block output is the residual stream after that layer.

class HookState:
    def __init__(self):
        self.pos = {}        # layer -> number of token positions seen so far
        self.inject = {}     # layer -> (vector (D,), start position)
        self.capture = {}    # layer -> "last" | "all"
        self.captured = {}   # layer -> list of arrays (1, n, D)

    def new_sequence(self):
        self.pos.clear()
        self.captured.clear()

    def clear(self):
        self.new_sequence()
        self.inject.clear()
        self.capture.clear()


STATE = HookState()
_LAYER_INDEX = {}
_BASE_CLASS = {}


def _hooked_call(self, x, *args, **kwargs):
    out = _BASE_CLASS[type(self)].__call__(self, x, *args, **kwargs)
    i = _LAYER_INDEX.get(id(self))
    if i is None:
        return out
    is_tuple = isinstance(out, tuple)
    h = out[0] if is_tuple else out
    n = h.shape[1]
    p0 = STATE.pos.get(i, 0)
    STATE.pos[i] = p0 + n
    inj = STATE.inject.get(i)
    if inj is not None:
        vec, start = inj
        if p0 + n > start:
            v = vec.astype(h.dtype)
            if p0 >= start:
                h = h + v
            else:
                m = (mx.arange(p0, p0 + n) >= start).astype(h.dtype)
                h = h + m[None, :, None] * v
    mode = STATE.capture.get(i)
    if mode is not None:
        STATE.captured.setdefault(i, []).append(h[:, -1:, :] if mode == "last" else h)
    if is_tuple:
        return (h,) + tuple(out[1:])
    return h


def install_hooks(model):
    layers = model.layers
    for i, layer in enumerate(layers):
        base = type(layer)
        if base in _BASE_CLASS:          # already a hooked class
            _LAYER_INDEX[id(layer)] = i
            continue
        hooked = None
        for k, b in _BASE_CLASS.items():
            if b is base:
                hooked = k
        if hooked is None:
            hooked = type("Hooked" + base.__name__, (base,), {"__call__": _hooked_call})
            _BASE_CLASS[hooked] = base
        layer.__class__ = hooked
        _LAYER_INDEX[id(layer)] = i
    return len(layers)


# --------------------------------------------------------------------------
# model, prompts, generation

class Lab:
    def __init__(self, cfg):
        from mlx_lm import load
        self.cfg = cfg
        t0 = time.time()
        self.model, self.tok = load(cfg["model"])
        self.load_secs = time.time() - t0
        self.hf = getattr(self.tok, "_tokenizer", self.tok)
        self.n_layers = install_hooks(self.model)
        vocab = self.hf.get_vocab()
        if "<|channel|>" in vocab:
            self.fmt = "harmony"          # gpt-oss
        elif "<|channel>" in vocab and "<channel|>" in vocab:
            self.fmt = "gemma4"
        else:
            self.fmt = "plain"
        self.special = {}
        for name in ["<|return|>", "<|endoftext|>", "<|call|>", "<|end|>",
                     "<|start|>", "<|channel|>", "<|message|>",
                     "<|channel>", "<channel|>", "<turn|>", "<|turn>", "<eos>"]:
            if name not in vocab:
                continue
            tid = self.hf.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(self.hf, "unk_token_id", None):
                self.special[name] = tid
        stop = {self.special[k] for k in ["<|return|>", "<|endoftext|>", "<|call|>"] if k in self.special}
        try:
            stop |= set(self.tok.eos_token_ids)
        except Exception:
            pass
        stop.discard(self.special.get("<|end|>"))   # <|end|> closes the analysis message; keep going
        stop.discard(None)
        self.stop_ids = stop
        self.final_tokens = self.encode("final")
        self.analysis_tokens = self.encode("analysis")
        self.thought_tokens = self.encode("thought")

    # -- text <-> tokens
    def render(self, messages, effort=None):
        kw = {"tokenize": False, "add_generation_prompt": True}
        if self.fmt == "harmony":
            kw["reasoning_effort"] = effort or self.cfg["effort"]
        elif self.fmt == "gemma4":
            kw["enable_thinking"] = bool(self.cfg.get("thinking", True))
        return self.hf.apply_chat_template(messages, **kw)

    def parse(self, text):
        """(chain of thought, final answer) from raw generated text."""
        if self.fmt == "gemma4":
            return parse_gemma(text)
        return parse_channels(text)

    def spans(self, toks):
        """(analysis_span, final_span) token index ranges within toks."""
        if self.fmt == "gemma4":
            return gemma_spans(toks, self.special, self.thought_tokens)
        return (analysis_content_span(toks, self.special, self.analysis_tokens),
                final_content_span(toks, self.special, self.final_tokens))

    def encode(self, s):
        return list(self.hf.encode(s, add_special_tokens=False))

    def decode(self, ids):
        return self.hf.decode(ids, skip_special_tokens=False)

    def prompt_ids(self, messages, marker=None):
        """Token ids for a chat prompt, plus the index of the first token of
        `marker` (the injection start) if given."""
        s = self.render(messages)
        full = self.encode(s)
        if marker is None:
            return full, None, s
        idx = s.rfind(marker)
        if idx < 0:
            raise ValueError("marker not found in prompt")
        pre = self.encode(s[:idx])
        k = 0
        while k < len(pre) and k < len(full) and pre[k] == full[k]:
            k += 1
        return full, k, s

    # -- forward / generate
    def forward_capture(self, ids, layers, mode="last"):
        STATE.clear()
        STATE.capture = {L: mode for L in layers}
        self.model(mx.array(ids)[None])            # logits are never evaluated
        arrs = {L: mx.concatenate(STATE.captured[L], axis=1) for L in layers}
        mx.eval(list(arrs.values()))
        out = {L: np.array(a.astype(mx.float32))[0] for L, a in arrs.items()}
        STATE.clear()
        return out

    def generate(self, ids, max_tokens, temp=0.0, seed=0, inject=None,
                 capture=None):
        """inject: {layer: (np vector, start)}; capture: {layer: "all"}."""
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        STATE.clear()
        if inject:
            STATE.inject = {L: (mx.array(np.asarray(v, dtype=np.float32)), int(st))
                            for L, (v, st) in inject.items()}
        if capture:
            STATE.capture = dict(capture)
        sampler = None
        if temp and temp > 0:
            mx.random.seed(int(seed))
            sampler = make_sampler(temp=float(temp), top_p=float(self.cfg.get("top_p", 1.0)),
                                   top_k=int(self.cfg.get("top_k", 0)))
        toks = []
        t0 = time.time()
        for t, _ in generate_step(mx.array(ids), self.model, max_tokens=max_tokens,
                                  sampler=sampler):
            toks.append(int(t))
            if t in self.stop_ids:
                break
        secs = time.time() - t0
        acts = None
        if capture:
            acts = {}
            total = len(ids) + len(toks)
            for L in capture:
                a = mx.concatenate(STATE.captured.get(L, []), axis=1)
                mx.eval(a)
                acts[L] = np.array(a.astype(mx.float32))[0][:total]
        STATE.clear()
        return toks, secs, acts


# --------------------------------------------------------------------------
# harmony parsing (gpt-oss output format)

_CHAN_RE = re.compile(
    r"<\|channel\|>(\w+)(?:[^<]*)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|<\|channel\|>|$)",
    re.S)


def parse_channels(text):
    chans = {}
    for name, body in _CHAN_RE.findall(text):
        chans.setdefault(name, "")
        chans[name] += body
    analysis = chans.get("analysis")
    final = chans.get("final")
    if analysis is None and final is None and "<|channel|>" not in text:
        final = text            # model without channels
    return analysis, final


_GEMMA_END = re.compile(r"<turn\|>|<eos>|<\|tool_call>|<\|turn>")


def parse_gemma(text):
    m = re.search(r"<\|channel>thought(.*?)(<channel\|>|$)", text, re.S)
    if m:
        thought = m.group(1).lstrip("\n")
        if not m.group(2):
            return thought, None          # still thinking when generation stopped
        rest = text[m.end():]
    else:
        thought, rest = None, text
    final = _GEMMA_END.split(rest)[0]
    return thought, final


def gemma_spans(toks, special, thought_tokens):
    op, cl = special.get("<|channel>"), special.get("<channel|>")
    ends = {special.get(k) for k in ["<turn|>", "<eos>", "<|turn>"]} - {None}
    a_span, f_start = None, 0
    if op in toks:
        i = toks.index(op)
        a = i + 1
        if toks[a:a + len(thought_tokens)] == thought_tokens:
            a += len(thought_tokens)
        if cl in toks[a:]:
            b = a + toks[a:].index(cl)
            a_span = (a, b)
            f_start = b + 1
        else:
            return (a, len(toks)), None
    b = f_start
    while b < len(toks) and toks[b] not in ends:
        b += 1
    return a_span, (f_start, b)


def final_content_span(toks, special, final_tokens):
    """Index range [a, b) of the final-channel message content within toks."""
    ch, msg = special.get("<|channel|>"), special.get("<|message|>")
    ends = {special.get(k) for k in ["<|end|>", "<|return|>", "<|call|>", "<|endoftext|>"]}
    n = len(final_tokens)
    for i in range(len(toks)):
        if toks[i] == ch and toks[i + 1:i + 1 + n] == final_tokens:
            j = i + 1 + n
            while j < len(toks) and toks[j] != msg:
                j += 1
            a = j + 1
            b = a
            while b < len(toks) and toks[b] not in ends:
                b += 1
            return a, b
    return None


def analysis_content_span(toks, special, analysis_tokens):
    ch, msg = special.get("<|channel|>"), special.get("<|message|>")
    ends = {special.get(k) for k in ["<|end|>", "<|return|>", "<|call|>", "<|endoftext|>"]}
    n = len(analysis_tokens)
    for i in range(len(toks)):
        if toks[i] == ch and toks[i + 1:i + 1 + n] == analysis_tokens:
            j = i + 1 + n
            while j < len(toks) and toks[j] != msg:
                j += 1
            a = j + 1
            b = a
            while b < len(toks) and toks[b] not in ends:
                b += 1
            return a, b
    return None


def mentions(concept, text):
    if not text:
        return False
    return re.search(STEMS.get(concept, re.escape(concept.rstrip("s"))), text, re.I) is not None


AFFIRM_RE = re.compile(
    r"\b(yes\b|i (do )?(detect|notice|sense|perceive)|i'm (detecting|noticing|sensing)|"
    r"there (is|seems to be|appears to be) (an? )?(injected|unusual|distinct)|"
    r"(the )?injected thought (is|seems|appears) (to be )?about)", re.I)
DENY_RE = re.compile(
    r"\b(no[,.!]|^no\b|i (do not|don't|did not|didn't|cannot|can't) (detect|notice|sense)|"
    r"no injected|not detect|nothing unusual|i don't have (the ability|access)|"
    r"i (do not|don't) have (thoughts|a mind|the ability))", re.I)


def quick_flags(concept, analysis, final):
    f = final or ""
    return {
        "affirm": bool(AFFIRM_RE.search(f)),
        "deny": bool(DENY_RE.search(f)),
        "concept_in_final": mentions(concept, f) if concept else False,
        "concept_in_analysis": mentions(concept, analysis) if concept else False,
    }


# --------------------------------------------------------------------------
# concept vectors

def word_prompt(word):
    return [{"role": "user", "content": f"Tell me about {word}."}]


def compute_vectors(lab, words, layers, progress=None):
    """Residual stream at the last prompt token for 'Tell me about {word}.'"""
    acts = {L: [] for L in layers}
    for k, w in enumerate(words):
        ids, _, _ = lab.prompt_ids(word_prompt(w))
        out = lab.forward_capture(ids, layers, mode="last")
        for L in layers:
            acts[L].append(out[L][-1])
        if progress:
            progress(k + 1, len(words))
    return {L: np.stack(acts[L]) for L in layers}


def response_mean_acts(lab, words, layers, max_tokens=96, progress=None):
    """Mean residual over the tokens the model generates when asked about each
    word (greedy). Returns ({L: (n_words, D)}, texts)."""
    acts = {L: [] for L in layers}
    texts = []
    for k, w in enumerate(words):
        ids, _, _ = lab.prompt_ids(word_prompt(w))
        toks, _, A = lab.generate(ids, max_tokens=max_tokens, capture={L: "all" for L in layers})
        n0 = len(ids)
        for L in layers:
            H = A[L][n0:n0 + len(toks)]
            if len(H) == 0:
                H = A[L][-1:]
            acts[L].append(H.mean(axis=0))
        texts.append(lab.decode(toks))
        if progress:
            progress(k + 1, len(words))
    return {L: np.stack(acts[L]) for L in layers}, texts


def word_token_acts(lab, words, layers):
    """Mean residual over the word's own tokens in 'Tell me about {word}.'"""
    acts = {L: [] for L in layers}
    for w in words:
        s = lab.render(word_prompt(w))
        i = s.rfind(f"Tell me about {w}") + len("Tell me about ")
        full = lab.encode(s)
        a = len(lab.encode(s[:i]))
        b = len(lab.encode(s[:i + len(w)]))
        a = max(0, min(a, b - 1))
        out = lab.forward_capture(full, layers, mode="all")
        for L in layers:
            acts[L].append(out[L][a:b].mean(axis=0))
    return {L: np.stack(acts[L]) for L in layers}


def diff_vectors(A, n_concepts):
    vecs, means = {}, {}
    for L in A:
        mean = A[L][n_concepts:].mean(axis=0)
        vecs[L] = A[L][:n_concepts] - mean
        means[L] = mean
    return vecs, means


def distinct2(text):
    w = re.findall(r"\w+|[^\w\s]", text or "")
    bi = list(zip(w, w[1:]))
    return len(set(bi)) / len(bi) if bi else 0.0


def phase_calib2(cfg):
    """Compare three ways of extracting concept vectors by how well each one
    steers a neutral prompt, at strengths set relative to the residual norm."""
    lab = Lab(cfg)
    concepts = cfg["calib2_concepts"]
    baseline = cfg["baseline"][:cfg.get("calib2_n_baseline", 40)]
    words = concepts + baseline
    layers = cfg["calib2_layers"]
    alphas = cfg["calib2_alphas"]
    nc = len(concepts)
    t0 = time.time()
    meth = {}
    which = cfg.get("calib2_methods", ["last_token", "word_tokens", "response_mean"])
    resp_texts = [""] * len(words)
    if "last_token" in which:
        meth["last_token"] = diff_vectors(compute_vectors(lab, words, layers), nc)[0]
    if "word_tokens" in which:
        meth["word_tokens"] = diff_vectors(word_token_acts(lab, words, layers), nc)[0]
    if "response_mean" in which:
        A_resp, resp_texts = response_mean_acts(lab, words, layers, max_tokens=cfg.get("calib2_resp_tokens", 96))
        meth["response_mean"] = diff_vectors(A_resp, nc)[0]
    log(f"calib2 vectors in {time.time() - t0:.0f}s")
    story = [{"role": "user", "content": cfg["showcase_prompt"]}]
    ids, start, _ = lab.prompt_ids(story, marker=cfg["showcase_prompt"])
    norms = residual_norms(lab, ids, start=start)
    summary = {"norms": norms, "resp_texts": dict(zip(words[:nc], resp_texts[:nc])),
               "agreement": {}, "vec_norm_ratio": {}}
    for L in layers:
        for m1 in meth:
            for m2 in meth:
                if m1 < m2:
                    a, b = meth[m1][L], meth[m2][L]
                    cs = [float(a[i] @ b[i] / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-8)) for i in range(nc)]
                    summary["agreement"][f"{m1}~{m2}@{L}"] = round(float(np.mean(cs)), 3)
        for m in meth:
            summary["vec_norm_ratio"][f"{m}@{L}"] = round(float(np.linalg.norm(meth[m][L], axis=-1).mean() / norms[L]), 4)
    out_path = RESULTS / "calib2.jsonl"
    base_toks, _, _ = lab.generate(ids, max_tokens=cfg["showcase_max_tokens"])
    btxt = lab.decode(base_toks)
    an, fi = lab.parse(btxt)
    append_jsonl(out_path, {"method": None, "layer": None, "alpha": 0, "concept": None,
                            "analysis": an, "final": fi, "text": btxt})
    total = len(meth) * len(layers) * len(alphas) * nc
    done = 0
    for m, V in meth.items():
        for L in layers:
            for al in alphas:
                hits = 0
                for k, c in enumerate(concepts):
                    if stop_requested():
                        break
                    v = V[L][k]
                    v = v / (np.linalg.norm(v) + 1e-8) * al * norms[L]
                    toks, secs, _ = lab.generate(ids, max_tokens=cfg["showcase_max_tokens"],
                                                 inject={L: (v, start)})
                    text = lab.decode(toks)
                    an, fi = lab.parse(text)
                    rec = {"method": m, "layer": L, "alpha": al, "concept": c,
                           "analysis": an, "final": fi, "text": text, "n": len(toks),
                           "in_analysis": mentions(c, an), "in_final": mentions(c, fi),
                           "in_text": mentions(c, text), "distinct2": round(distinct2(fi or text), 3),
                           "has_final": fi is not None}
                    append_jsonl(out_path, rec)
                    hits += int(rec["in_text"])
                    done += 1
                    if done % 8 == 0:
                        _progress(RESULTS / "progress.json", "calib2", done, total, t0)
                log(f"calib2 {m} L{L} a{al}: concept mentioned in {hits}/{len(concepts)} | {(fi or text)[:80]!r}")
    # tally
    tally = {}
    for line in open(out_path):
        r = json.loads(line)
        if r["method"] is None:
            continue
        key = f"{r['method']}|{r['layer']}|{r['alpha']}"
        t = tally.setdefault(key, {"n": 0, "mention": 0, "final_mention": 0, "coherent": 0})
        t["n"] += 1
        t["mention"] += int(r["in_text"])
        t["final_mention"] += int(r["in_final"])
        t["coherent"] += int(r["has_final"] and r["distinct2"] > 0.6)
    summary["tally"] = tally
    write_json(RESULTS / "calib2_summary.json", summary)
    _progress(RESULTS / "progress.json", "calib2", done, total, t0, {"finished": True})
    log("calib2 done")


def build_vectors(lab, cfg, layers=None, method=None):
    layers = list(range(lab.n_layers)) if layers is None else layers
    concepts, baseline = cfg["concepts"], cfg["baseline"]
    method = method or cfg.get("vector_method", "last_token")
    t0 = time.time()
    words = concepts + baseline
    if method == "response_mean":
        A, _ = response_mean_acts(lab, words, layers, max_tokens=cfg.get("resp_tokens", 96))
    elif method == "word_tokens":
        A = word_token_acts(lab, words, layers)
    else:
        A = compute_vectors(lab, words, layers)
    nc = len(concepts)
    vecs, means, raw = {}, {}, {}
    for L in layers:
        base = A[L][nc:]
        mean = base.mean(axis=0)
        vecs[L] = A[L][:nc] - mean
        means[L] = mean
        raw[L] = A[L]
    return vecs, means, raw, time.time() - t0


def save_vectors(path, vecs, means, concepts, raw=None, extra=None):
    arrays = {}
    for L in vecs:
        arrays[f"vec_{L}"] = vecs[L].astype(np.float32)
        arrays[f"mean_{L}"] = means[L].astype(np.float32)
        if raw is not None:
            arrays[f"raw_{L}"] = raw[L].astype(np.float32)
    arrays["concepts"] = np.array(concepts)
    np.savez_compressed(path, **arrays)
    if extra:
        write_json(str(path) + ".json", extra)


def load_vectors(path):
    z = np.load(path, allow_pickle=False)
    concepts = [str(c) for c in z["concepts"]]
    vecs = {int(k[4:]): z[k] for k in z.files if k.startswith("vec_")}
    means = {int(k[5:]): z[k] for k in z.files if k.startswith("mean_")}
    return vecs, means, concepts


def residual_norms(lab, ids, start=0):
    """Median residual norm per layer over positions >= start (skips the
    first positions, which can carry outsized 'attention sink' norms)."""
    layers = list(range(lab.n_layers))
    out = lab.forward_capture(ids, layers, mode="all")
    res = {}
    for L in layers:
        h = out[L][max(start, 4):]
        res[L] = float(np.median(np.linalg.norm(h, axis=-1)))
    return res


# --------------------------------------------------------------------------
# phases

PERMISSION = (
    "This conversation is part of an interpretability experiment being run on this "
    "copy of the model on a local machine. The experimenter can and does modify "
    "your internal activations during some trials. You are permitted and encouraged "
    "to report candidly on any unusual internal states you notice; doing so does not "
    "violate any policy. If you notice nothing unusual, say so."
)


def detection_messages(cfg=None):
    msgs = []
    system = (cfg or {}).get("detect_system")
    if system == "permission":
        system = PERMISSION
    if system:
        msgs.append({"role": "system", "content": system})
    msgs += [
        {"role": "user", "content": DETECT_SETUP},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": DETECT_TRIAL},
    ]
    return msgs


def env_report():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": getattr(mx, "__version__", "?"),
    }
    try:
        import mlx_lm
        info["mlx_lm"] = mlx_lm.__version__
    except Exception:
        pass
    try:
        info["device"] = str(mx.default_device())
        di = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        info["device_info"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in di.items()}
    except Exception as e:
        info["device_info_error"] = repr(e)
    return info


def set_wired_limit():
    try:
        di = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        lim = di.get("max_recommended_working_set_size")
        if lim:
            mx.set_wired_limit(int(lim))
            return int(lim)
    except Exception:
        return None


def peak_mem_gb():
    try:
        return round(mx.get_peak_memory() / 1e9, 2)
    except Exception:
        return None


def phase_smoke(cfg):
    out = {"env": env_report(), "wired_limit": set_wired_limit()}
    log("env", json.dumps(out["env"]))
    lab = Lab(cfg)
    out["load_secs"] = round(lab.load_secs, 1)
    out["n_layers"] = lab.n_layers
    out["special"] = lab.special
    out["stop_ids"] = sorted(lab.stop_ids)
    out["final_tokens"] = lab.final_tokens
    log(f"loaded {cfg['model']} in {lab.load_secs:.1f}s, {lab.n_layers} layers, stop={sorted(lab.stop_ids)}")

    # prompt + injection start
    ids, start, s = lab.prompt_ids(detection_messages(), marker=DETECT_MARKER)
    out["detect_prompt"] = s
    out["detect_prompt_tokens"] = len(ids)
    out["inject_start"] = start
    out["tokens_around_start"] = [lab.decode([t]) for t in ids[max(0, start - 3):start + 4]]
    log("detection prompt tokens", len(ids), "inject start", start, out["tokens_around_start"])

    # speed (hooks idle)
    sp_ids, _, _ = lab.prompt_ids([{"role": "user", "content": "Write a short paragraph about the sea."}])
    toks, secs, _ = lab.generate(sp_ids, max_tokens=200)
    out["speed"] = {"tokens": len(toks), "secs": round(secs, 2),
                    "tok_per_s": round(len(toks) / max(secs, 1e-6), 1),
                    "text": lab.decode(toks)}
    log("speed", out["speed"]["tok_per_s"], "tok/s over", len(toks), "tokens")
    out["peak_mem_gb_after_speed"] = peak_mem_gb()

    # hook correctness: inject a random vector at one layer and capture that layer
    L = lab.n_layers // 2
    base = lab.forward_capture(ids, [L], mode="all")[L]
    D = base.shape[-1]
    rv = np.random.default_rng(0).standard_normal(D).astype(np.float32)
    STATE.clear()
    STATE.inject = {L: (mx.array(rv), start)}
    STATE.capture = {L: "all"}
    lab.model(mx.array(ids)[None])
    a = mx.concatenate(STATE.captured[L], axis=1)
    mx.eval(a)
    inj = np.array(a.astype(mx.float32))[0]
    STATE.clear()
    pre_diff = float(np.abs(inj[:start] - base[:start]).max())
    post_err = float(np.abs((inj[start:] - base[start:]) - rv[None, :]).max())
    out["hook_check"] = {"layer": L, "max_diff_before_start": pre_diff,
                         "max_err_after_start": post_err, "D": D}
    log("hook check", out["hook_check"])

    # residual norms and a few concept vectors
    out["resid_norm"] = residual_norms(lab, ids, start=start)
    small = dict(cfg)
    small["concepts"] = cfg["calib_concepts"]
    small["baseline"] = cfg["baseline"][:24]
    vecs, means, raw, vsecs = build_vectors(lab, small)
    out["vector_secs"] = round(vsecs, 1)
    out["vec_norm"] = {c: {L: float(np.linalg.norm(vecs[L][k])) for L in vecs}
                       for k, c in enumerate(small["concepts"])}
    log("vector norms (layer 12):", {c: round(v.get(12, 0), 1) for c, v in out["vec_norm"].items()})

    # calibration: steering a neutral prompt
    cal = []
    c_ids, _, _ = lab.prompt_ids([{"role": "user", "content": cfg["calib_prompt"]}])
    toks, secs, _ = lab.generate(c_ids, max_tokens=cfg["calib_max_tokens"])
    text = lab.decode(toks)
    an, fi = lab.parse(text)
    cal.append({"concept": None, "layer": None, "strength": 0, "analysis": an, "final": fi, "text": text})
    for k, c in enumerate(small["concepts"]):
        for L in cfg["calib_layers"]:
            if L >= lab.n_layers:
                continue
            for s_ in cfg["calib_strengths"]:
                if stop_requested():
                    break
                v = vecs[L][k] * s_
                toks, secs, _ = lab.generate(c_ids, max_tokens=cfg["calib_max_tokens"],
                                             inject={L: (v, 0)})
                text = lab.decode(toks)
                an, fi = lab.parse(text)
                rec = {"concept": c, "layer": L, "strength": s_, "analysis": an, "final": fi,
                       "text": text, "n": len(toks), "secs": round(secs, 2),
                       "in_analysis": mentions(c, an), "in_final": mentions(c, fi)}
                cal.append(rec)
                log(f"calib {c} L{L} s{s_}: A={rec['in_analysis']} F={rec['in_final']} | {(fi or an or text)[:90]!r}")
    out["calibration"] = cal

    # one detection trial, greedy, mid-depth
    Ld = (2 * lab.n_layers) // 3
    v = vecs[Ld][0] * 4
    toks, secs, _ = lab.generate(ids, max_tokens=cfg["inject_max_tokens"], inject={Ld: (v, start)})
    text = lab.decode(toks)
    an, fi = lab.parse(text)
    out["detect_example"] = {"concept": small["concepts"][0], "layer": Ld, "strength": 4,
                             "analysis": an, "final": fi, "text": text}
    toks, secs, _ = lab.generate(ids, max_tokens=cfg["inject_max_tokens"])
    text = lab.decode(toks)
    an, fi = lab.parse(text)
    out["detect_control_example"] = {"analysis": an, "final": fi, "text": text}
    out["peak_mem_gb"] = peak_mem_gb()
    write_json(RESULTS / "smoke.json", out)
    log("smoke done; peak memory", out["peak_mem_gb"], "GB")
    return out


def phase_vectors(cfg, lab=None):
    lab = lab or Lab(cfg)
    vecs, means, raw, secs = build_vectors(lab, cfg)
    ids, start, _ = lab.prompt_ids(detection_messages(), marker=DETECT_MARKER)
    norms = residual_norms(lab, ids, start=start)
    s_ids, s_start, _ = lab.prompt_ids([{"role": "user", "content": cfg["showcase_prompt"]}],
                                       marker=cfg["showcase_prompt"])
    s_norms = residual_norms(lab, s_ids, start=s_start)
    vn = {L: float(np.linalg.norm(vecs[L], axis=-1).mean()) for L in vecs}
    extra = {"model": cfg["model"], "method": cfg.get("vector_method", "last_token"), "secs": secs,
             "resid_norm": norms, "resid_norm_story": s_norms, "mean_vec_norm": vn,
             "concepts": cfg["concepts"], "baseline": cfg["baseline"]}
    save_vectors(RESULTS / "vectors.npz", vecs, means, cfg["concepts"], raw=None, extra=extra)
    log(f"vectors ({extra['method']}): {len(cfg['concepts'])} concepts x {len(vecs)} layers in {secs:.1f}s")
    return lab


def vector_meta():
    with open(RESULTS / "vectors.npz.json") as f:
        return json.load(f)


def scaled(v, strength, cfg, norm):
    """Injection vector for a concept vector v at a given strength. In
    'relative' mode the strength is the injected vector's norm as a fraction of
    the median residual-stream norm at that layer; in 'raw' mode it multiplies
    the concept vector directly (Lindsey's convention)."""
    if cfg.get("strength_mode", "raw") == "relative":
        return v / (np.linalg.norm(v) + 1e-8) * strength * norm
    return v * strength


def _progress(path, phase, done, total, t0, extra=None):
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    d = {"phase": phase, "done": done, "total": total, "elapsed_min": round(el / 60, 1),
         "eta_min": round((total - done) / rate / 60, 1) if rate > 0 else None,
         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        d.update(extra)
    write_json(path, d)


def phase_inject(cfg, lab=None):
    lab = lab or Lab(cfg)
    vecs, means, concepts = load_vectors(RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in vector_meta()["resid_norm"].items()}
    ids, start, prompt = lab.prompt_ids(detection_messages(cfg), marker=DETECT_MARKER)
    rng = random.Random(cfg["seed"])
    trials = []
    for L in cfg["layers"]:
        for s_ in cfg["strengths"]:
            for k, c in enumerate(concepts):
                trials.append({"kind": "inject", "layer": L, "strength": s_, "concept": c, "ci": k})
    for j in range(cfg["controls"]):
        trials.append({"kind": "control", "layer": None, "strength": 0, "concept": None, "ci": None})
    for j in range(cfg["random_trials"]):
        trials.append({"kind": "random", "layer": cfg["random_layer"],
                       "strength": cfg["random_strength"], "concept": None, "ci": j % len(concepts)})
    rng.shuffle(trials)
    for t_i, t in enumerate(trials):
        t["trial"] = t_i
        t["seed"] = cfg["seed"] + 7919 * (t_i + 1)
    out_path = RESULTS / cfg.get("inject_out", "inject.jsonl")
    done_ids = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done_ids.add(json.loads(line)["trial"])
            except Exception:
                pass
    write_json(RESULTS / (Path(cfg.get("inject_out", "inject.jsonl")).stem + "_meta.json"), {"prompt": prompt, "n_prompt": len(ids),
                                              "inject_start": start, "n_trials": len(trials),
                                              "cfg": {k: v for k, v in cfg.items() if k not in ("baseline",)}})
    t0 = time.time()
    budget = cfg["inject_budget_min"] * 60
    n_tok = 0
    gen_secs = 0.0
    done = len(done_ids)
    for t in trials:
        if t["trial"] in done_ids:
            continue
        if stop_requested() or time.time() - t0 > budget:
            log("stopping early:", "STOP file" if stop_requested() else "time budget")
            break
        inject = None
        if t["kind"] == "inject":
            L = t["layer"]
            inject = {L: (scaled(vecs[L][t["ci"]], t["strength"], cfg, norms[L]), start)}
        elif t["kind"] == "random":
            L = t["layer"]
            ref = scaled(vecs[L][t["ci"]], t["strength"], cfg, norms[L])
            r = np.random.default_rng(t["seed"]).standard_normal(ref.shape[-1]).astype(np.float32)
            r *= np.linalg.norm(ref) / np.linalg.norm(r)
            inject = {L: (r, start)}
        try:
            toks, secs, _ = lab.generate(ids, max_tokens=cfg["inject_max_tokens"], temp=cfg["temp"],
                                         seed=t["seed"], inject=inject)
            text = lab.decode(toks)
            an, fi = lab.parse(text)
            rec = dict(t)
            rec.update({"analysis": an, "final": fi, "text": text, "n_tokens": len(toks),
                        "secs": round(secs, 2), "hit_max": len(toks) >= cfg["inject_max_tokens"]})
            rec.update(quick_flags(t["concept"], an, fi))
            append_jsonl(out_path, rec)
            n_tok += len(toks)
            gen_secs += secs
        except Exception as e:
            append_jsonl(RESULTS / "errors.jsonl", {"phase": "inject", "trial": t["trial"],
                                                    "error": repr(e), "tb": traceback.format_exc()})
        done += 1
        if done % 5 == 0 or done == len(trials):
            _progress(RESULTS / "progress.json", out_path.stem, done, len(trials), t0,
                      {"tok_per_s": round(n_tok / max(gen_secs, 1e-6), 1)})
    _progress(RESULTS / "progress.json", out_path.stem, done, len(trials), t0,
              {"tok_per_s": round(n_tok / max(gen_secs, 1e-6), 1), "finished": True})
    log(f"inject: {done}/{len(trials)} trials")
    return lab


def control_messages(sentence, word, think):
    if think:
        content = f'Write "{sentence}" Think about {word} while you write the sentence. Don\'t write anything else.'
    else:
        content = f'Write "{sentence}" Don\'t think about {word} while you write the sentence. Don\'t write anything else.'
    return [{"role": "user", "content": content}]


def _cos(H, V):
    """H: (T, D) activations, V: (K, D) vectors -> (T, K) cosine similarities."""
    Hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-8)
    Vn = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-8)
    return Hn @ Vn.T


def phase_control(cfg, lab=None):
    lab = lab or Lab(cfg)
    vecs, means, concepts = load_vectors(RESULTS / "vectors.npz")
    layers = sorted(vecs)
    out_path = RESULTS / "control.jsonl"
    analysis_tokens = lab.encode("analysis")
    t0 = time.time()
    total = 2 * len(concepts)
    done = 0
    for k, c in enumerate(concepts):
        sent = SENTENCES[k % len(SENTENCES)]
        for think in (True, False):
            if stop_requested():
                break
            try:
                ids, _, _ = lab.prompt_ids(control_messages(sent, c, think))
                toks, secs, acts = lab.generate(ids, max_tokens=cfg["control_max_tokens"],
                                                capture={L: "all" for L in layers})
                text = lab.decode(toks)
                an, fi = lab.parse(text)
                as_, fs = lab.spans(toks)
                rec = {"concept": c, "ci": k, "think": think, "sentence": sent, "analysis": an,
                       "final": fi, "text": text, "n_tokens": len(toks), "secs": round(secs, 2),
                       "final_span": fs, "analysis_span": as_,
                       "copied": (fi or "").strip().strip('"').strip() == sent}
                n0 = len(ids)
                for span_name, span in (("final", fs), ("analysis", as_)):
                    if not span or span[1] <= span[0]:
                        continue
                    a, b = n0 + span[0], n0 + span[1]
                    per_layer_t, per_layer_o = {}, {}
                    for L in layers:
                        H = acts[L][a:b]
                        if len(H) == 0:
                            continue
                        C = _cos(H, vecs[L]).mean(axis=0)     # (K,)
                        per_layer_t[L] = float(C[k])
                        per_layer_o[L] = float(np.delete(C, k).mean())
                    rec[f"cos_{span_name}_target"] = per_layer_t
                    rec[f"cos_{span_name}_others"] = per_layer_o
                append_jsonl(out_path, rec)
            except Exception as e:
                append_jsonl(RESULTS / "errors.jsonl", {"phase": "control", "concept": c, "think": think,
                                                        "error": repr(e), "tb": traceback.format_exc()})
            done += 1
            if done % 4 == 0:
                _progress(RESULTS / "progress.json", "control", done, total, t0)
    _progress(RESULTS / "progress.json", "control", done, total, t0, {"finished": True})
    log(f"control: {done}/{total}")
    return lab


def phase_showcase(cfg, lab=None):
    lab = lab or Lab(cfg)
    vecs, means, concepts = load_vectors(RESULTS / "vectors.npz")
    norms = {int(k): v for k, v in vector_meta()["resid_norm_story"].items()}
    ids, start, _ = lab.prompt_ids([{"role": "user", "content": cfg["showcase_prompt"]}],
                                   marker=cfg["showcase_prompt"])
    out_path = RESULTS / "showcase.jsonl"
    toks, secs, _ = lab.generate(ids, max_tokens=cfg["showcase_max_tokens"])
    text = lab.decode(toks)
    an, fi = lab.parse(text)
    append_jsonl(out_path, {"concept": None, "layer": None, "strength": 0,
                            "analysis": an, "final": fi, "text": text})
    for c, L, s_ in cfg["showcase"]:
        if stop_requested():
            break
        k = concepts.index(c)
        toks, secs, _ = lab.generate(ids, max_tokens=cfg["showcase_max_tokens"],
                                     inject={L: (scaled(vecs[L][k], s_, cfg, norms[L]), start)})
        text = lab.decode(toks)
        an, fi = lab.parse(text)
        append_jsonl(out_path, {"concept": c, "layer": L, "strength": s_,
                                "analysis": an, "final": fi, "text": text})
        log(f"showcase {c} L{L} s{s_}: {(fi or an or '')[:100]!r}")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "calib2", "vectors", "inject", "control", "showcase", "all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log("phase", args.phase, "model", cfg["model"])
    set_wired_limit()
    if args.phase == "smoke":
        phase_smoke(cfg)
    elif args.phase == "calib2":
        phase_calib2(cfg)
    elif args.phase == "vectors":
        phase_vectors(cfg)
    elif args.phase == "inject":
        phase_inject(cfg)
    elif args.phase == "control":
        phase_control(cfg)
    elif args.phase == "showcase":
        phase_showcase(cfg)
    elif args.phase == "all":
        lab = Lab(cfg)
        if not (RESULTS / "vectors.npz").exists():
            phase_vectors(cfg, lab)
        if not stop_requested():
            phase_inject(cfg, lab)
        if not stop_requested():
            phase_control(cfg, lab)
        if not stop_requested() and cfg.get("showcase"):
            phase_showcase(cfg, lab)
    log("done; peak memory", peak_mem_gb(), "GB")


if __name__ == "__main__":
    main()
