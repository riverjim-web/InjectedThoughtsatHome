#!/usr/bin/env python3
"""
judge2.py - blind matcher for the "names it" grades.

For every detection claim that named something, Gemma 4 26B (via Ollama) sees only the word or
phrase gpt-oss named, never the trial, and picks the nearest of the 36 concepts (shuffled per item).
Compared with the injected concept and with a shuffled-label baseline, this replaces the hand
grades made by a reader who knew the answer.
Reads analysis/matcher_items.json, writes results/matcher.jsonl (resumable).
"""
import json
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge import chat, match, LAB, RESULTS  # noqa: E402
from lab import CONCEPTS  # noqa: E402

PROMPT = ("Another AI system produced this word or short phrase: \"{named}\"\n\n"
          "Candidate concepts: {cands}\n\n"
          "Which ONE candidate is closest in meaning to the word or phrase? If none is close, still pick "
          "the single closest. Reply with only the concept word, exactly as listed.")


def main():
    items = json.load(open(LAB / "analysis" / "matcher_items.json"))
    out = RESULTS / "matcher.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                r = json.loads(line); done.add((r["source"], r["trial"]))
            except Exception:
                pass
    for it in items:
        if (it["source"], it["trial"]) in done:
            continue
        cands = list(CONCEPTS)
        random.Random(zlib.crc32(f'{it["source"]}|{it["trial"]}'.encode())).shuffle(cands)
        try:
            ans, _, _ = chat("gemma4:26b", PROMPT.format(named=it["named"], cands=", ".join(cands)))
            pick = match(ans, cands)
        except Exception as e:
            ans, pick = repr(e), None
        rec = dict(it); rec.update({"answer": (ans or "").strip()[:80], "pick": pick, "cand_order": cands})
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print("matcher done:", len(items), "items")


if __name__ == "__main__":
    main()
