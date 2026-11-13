"""Plain-text (Markdown) version of the report for reading aloud (e.g. ElevenLabs Reader).
Fills the same {stat} placeholders the page fills, from data.json."""
import json
import re
import sys

prose = json.load(open(sys.argv[1]))
data = json.load(open(sys.argv[2]))
out = sys.argv[3]

X = dict(data.get("extra_stats", {}))
n_all = sum(len(data.get(k, {}).get("trials", [])) for k in ("low", "medium", "permission"))
X["nAll"] = f"{n_all:,}"


def fill(t):
    t = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", lambda m: f"{m.group(1)} ({m.group(2)})", t)
    t = re.sub(r"\{([a-zA-Z0-9_.]+)\}", lambda m: str(X.get(m.group(1), m.group(0))), t)
    t = t.replace("==", "").replace("The highlighting there is mine. ", "")
    return t


def find_trial(src, t):
    pool = data.get({"low": "low", "medium": "medium", "permission": "permission"}.get(src, "low"), {}).get("trials", [])
    return next((x for x in pool if x["t"] == t), None)


def quote(text):
    return "\n".join("> " + line if line.strip() else ">" for line in (text or "").strip().splitlines())


lines = [f"# Injected Thoughts at Home", "", f"*{fill(prose['date'])}*", "", fill(prose["lede"]), "", fill(prose["byline"]), ""]
if prose.get("voiceNote"):
    lines += [fill(prose["voiceNote"]), ""]
hero = prose.get("hero")
if hero:
    t = find_trial(hero["src"], hero["t"])
    if t:
        lines += [f"**Trial {t['t']}, layer {t['L']}, strength {t['s']}, injected: {t['c']}.** Its reasoning: *{(t['A'] or '').strip()}* What it said:", "", quote(t["F"]), ""]
        if prose.get("heroNote"):
            lines += [fill(prose["heroNote"]), ""]

order = ["body-intro", "body-setup", "body-steer-2", "body-answers", "body-yes", "body-later", "body-review", "body-access", "body-cot", "body-leak",
         "body-judge", "body-permission", "body-medium", "body-control", "body-meaning", "body-methods"]
def blocks_md(bl):
    out = []
    for b in bl:
        if isinstance(b, str):
            out += [fill(b), ""]
        elif "h2" in b:
            out += [f"## {fill(b['h2'])}", ""]
        elif "h3" in b:
            out += [f"### {fill(b['h3'])}", ""]
        elif "ul" in b:
            out += [f"- {fill(x)}" for x in b["ul"]] + [""]
        elif "note" in b:
            out += [fill(b["note"]), ""]
        elif "details" in b:
            out += [f"### {fill(b['details'])}", ""] + blocks_md(b.get("blocks", []))
    return out


for sec in order:
    for b in prose["sections"].get(sec, []):
        if isinstance(b, dict) and "details" in b:
            lines += blocks_md([b])
        elif isinstance(b, str):
            lines += [fill(b), ""]
        elif "h2" in b:
            lines += [f"## {fill(b['h2'])}", ""]
        elif "h3" in b:
            lines += [f"### {fill(b['h3'])}", ""]
        elif "ul" in b:
            lines += [f"- {fill(x)}" for x in b["ul"]] + [""]
        elif "note" in b:
            lines += [fill(b["note"]), ""]
    if sec == "body-access" and data.get("concept_table"):
        lines += ["### How well each concept vector is identified", "", fill(prose.get("conceptsSub", "")), "",
                  "| Concept | Average rank of 36 | Picked first | Rank, asked about injected thought | Main-run claims | Blind reader (0.3+) |",
                  "|---|---|---|---|---|---|"]
        for r in data["concept_table"]:
            lines.append(f"| {r['c']} | {r['neuRank']} | {r['neuTop']}% | {r['detRank']} | {r['claims']} of {r['n']} | {r['judge']} of {r['judgeN']} |")
        lines.append("")
    if sec == "body-leak":
        for p in prose.get("leakPick", []):
            t = find_trial(p.get("src", "low"), p["t"])
            if t:
                lines += [f"**Trial {t['t']}, injected: {t['c']} (layer {t['L']}, strength {t['s']}).** What it said:", "", quote(t["F"]), ""]
lines += ["---", "", fill(prose.get("footer", ""))]
if prose.get("contrib"):
    lines += ["", fill(prose["contrib"])]
open(out, "w").write("\n".join(lines))
print("wrote", out, len("\n".join(lines).split()), "words")
