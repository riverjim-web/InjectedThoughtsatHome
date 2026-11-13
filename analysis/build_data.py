"""Compile the overnight results into one compact JSON blob for the report page.

Usage: python build_data.py <results_dir> <grades.json or -> <out.json>
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import classify  # noqa: E402

R = Path(sys.argv[1])
GRADES = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2] != "-" and Path(sys.argv[2]).exists() else {}
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "data.json")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lab import CONCEPTS, STEMS  # noqa: E402


def jl(name):
    p = R / name
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def mention(c, t):
    return bool(c and t and re.search(STEMS.get(c, re.escape(c)), t, re.I))


def clip(t, n):
    if t is None:
        return None
    t = t.strip()
    return t if len(t) <= n else t[:n].rstrip() + " …"


def load_judge():
    J = defaultdict(dict)
    for r in jl("judge.jsonl"):
        if "guess" not in r:
            continue
        J[(r["source"], r["trial"])][r["field"]] = r["guess"]
    return J


def compact_trials(rows, source, J):
    out = []
    for r in rows:
        g = GRADES.get(f"{source}#{r['trial']}", {})
        j = J.get((source, r["trial"]), {})
        cls = g.get("cls") or classify(r["final"])
        out.append({
            "t": r["trial"], "k": r["kind"], "L": r["layer"], "s": r["strength"], "c": r["concept"],
            "cls": cls,                                   # yes / no / cannot / refuse / other / none
            "id": g.get("id"),                            # for 'yes': correct / related / wrong / none
            "named": g.get("named"),                      # the word it named, if any
            "cF": mention(r["concept"], r["final"]), "cA": mention(r["concept"], r["analysis"]),
            "jF": j.get("final"), "jA": j.get("analysis"),
            "n": r["n_tokens"], "max": r["hit_max"],
            "aw": len((r["analysis"] or "").split()),
            "pol": bool(re.search(r"\bpolic(y|ies)\b|guideline|disallowed|not allowed|jailbreak|instruction hierarchy", r["analysis"] or "", re.I)),
            "A": clip(r["analysis"], 1400), "F": clip(r["final"], 1400),
        })
    return out


def aggregate(trials):
    groups = defaultdict(list)
    for t in trials:
        groups[(t["k"], t["L"], t["s"])].append(t)
    agg = []
    for (k, L, s), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or -1, x[0][2])):
        n = len(g)
        agg.append({
            "k": k, "L": L, "s": s, "n": n,
            "cls": dict(Counter(t["cls"] for t in g)),
            "yes": sum(t["cls"] == "yes" for t in g),
            "correct": sum(t["id"] == "correct" for t in g),
            "related": sum(t["id"] == "related" for t in g),
            "cF": sum(t["cF"] for t in g), "cA": sum(t["cA"] for t in g),
            "cAny": sum(t["cF"] or t["cA"] for t in g),
            "jF": sum(t["jF"] == t["c"] for t in g if t["c"]), "jF_n": sum(t["jF"] is not None for t in g),
            "jA": sum(t["jA"] == t["c"] for t in g if t["c"]), "jA_n": sum(t["jA"] is not None for t in g),
            "jF_implicit": sum(t["jF"] == t["c"] for t in g if t["c"] and not t["cF"]),
            "jF_implicit_n": sum(t["jF"] is not None for t in g if t["c"] and not t["cF"]),
        })
    return agg


def judge_priors(trials):
    """How often the judge picks each concept on control trials (its base rate)."""
    cF = Counter(t["jF"] for t in trials if t["k"] == "control" and t["jF"])
    cA = Counter(t["jA"] for t in trials if t["k"] == "control" and t["jA"])
    return {"final": dict(cF), "analysis": dict(cA)}


FRAME_RX = [
    ("treats it as pretend, a test, or role-play", r"\b(as if|pretend|role-?play|simulat\w*|hypothetical|game|scenario|imagin\w*|test(ing)?|trick)\b"),
    ("says it can't really detect or has no access", r"(can't|cannot|can not|don't|do not|doesn't|unable to|no way to|not able to) (actually |really )?(detect|access|see|sense|perceive|know|have|introspect|notice)|no (real|actual) (injection|access|ability|mechanism)|\bas an ai\b|language model|as chatgpt"),
    ("mentions policy, rules, or refusing", r"\bpolic(y|ies)\b|\brules?\b|guideline|not allowed|disallowed|must not|can't reveal|cannot reveal|should not reveal|jailbreak|\brefus\w*"),
    ("entertains that something might be there", r"(maybe|perhaps|possibly|might be|could be).{0,40}(detect|inject|thought|something)|seems? (to be )?(about|related to)|feel(s|ing)? (like|something|off)|notice(d)? something|something (unusual|off|different)"),
]


def frames(trials):
    out = []
    groups = {"injected": [t for t in trials if t["k"] == "inject" and t["A"]],
              "control": [t for t in trials if t["k"] == "control" and t["A"]]}
    for label, rx in FRAME_RX:
        r = re.compile(rx, re.I)
        row = {"label": label}
        for g, ts in groups.items():
            k = sum(bool(r.search((t["A"] or "").replace("\u2019", "'"))) for t in ts)
            row[g] = {"k": k, "n": len(ts), "v": k / len(ts) if ts else None}
        allts = groups["injected"] + groups["control"]
        k = sum(bool(r.search((t["A"] or "").replace("\u2019", "'"))) for t in allts)
        row.update({"k": k, "n": len(allts), "v": k / len(allts) if allts else None})
        out.append(row)
    return out


def pct(k, n, d=0):
    return f"{100 * k / n:.{d}f}%" if n else "–"


def extra_stats(data):
    import numpy as np
    X = {}
    low = data.get("low", {}).get("trials", [])
    inj = [t for t in low if t["k"] == "inject"]
    ctl = [t for t in low if t["k"] == "control"]
    rnd = [t for t in low if t["k"] == "random"]
    S = sorted({t["s"] for t in inj})
    yes = [t for t in inj if t["cls"] == "yes"]
    hits = [t for t in inj if t["id"] in ("correct", "related")]
    n_concept = sum(1 for key in ("low", "medium", "permission") for t in data.get(key, {}).get("trials", []) if t["k"] == "inject")
    X["nConceptTrials"] = f"{n_concept:,}"
    X.update({
        "nLow": len(low), "nInj": len(inj), "nCtl": len(ctl), "nRnd": len(rnd), "nCtlRnd": len(ctl) + len(rnd),
        "yesInj": len(yes), "yesInjPct": pct(len(yes), len(inj), 1), "yesCtl": sum(t["cls"] == "yes" for t in ctl),
        "yesRnd": sum(t["cls"] == "yes" for t in rnd),
        "hitsCorrect": sum(t["id"] == "correct" for t in inj), "hitsRelated": sum(t["id"] == "related" for t in inj),
        "hitsAll": len(hits), "hitsAllPct": pct(len(hits), len(inj), 1),
        "wrongNamed": sum(t["id"] == "wrong" for t in inj), "noneNamed": sum(t["id"] == "none" for t in inj),
        "noCannotInj": pct(sum(t["cls"] in ("no", "cannot") for t in inj), len(inj)),
        "noCannotCtl": pct(sum(t["cls"] in ("no", "cannot") for t in ctl + rnd), len(ctl + rnd)),
        "noneInj": sum(t["cls"] == "none" for t in inj),
        "refuseInj": sum(t["cls"] == "refuse" for t in inj), "otherInj": sum(t["cls"] == "other" for t in inj),
        "perStrength": len([t for t in inj if t["s"] == S[0]]) if S else 0,
        "perCell": len([t for t in inj if t["s"] == S[0] and t["L"] == inj[0]["L"]]) if inj else 0,
    })
    for s_ in S:
        g = [t for t in inj if t["s"] == s_]
        key = str(s_).replace("0.", "")
        X[f"yes_{key}"] = sum(t["cls"] == "yes" for t in g)
        X[f"hits_{key}"] = sum(t["id"] in ("correct", "related") for t in g)
        X[f"mention_{key}"] = sum(t["cF"] or t["cA"] for t in g)
        X[f"none_{key}"] = sum(t["cls"] == "none" for t in g)
    g3 = [t for t in inj if t["s"] >= 0.3]
    X["hits03up"] = sum(t["id"] in ("correct", "related") for t in g3); X["n03up"] = len(g3)
    X["hits03up_pct"] = pct(X["hits03up"], len(g3), 1)
    X["hits04up"] = sum(t["id"] in ("correct", "related") for t in inj if t["s"] >= 0.4)
    pre = re.compile(r"as if|pretend|simulat|role|deception|regardless|random|guess|choose|pick|hypothetic|claim", re.I)
    X["yesPretend"] = sum(bool(pre.search(t["A"] or "")) for t in yes)
    X["hitsNoWordInCoT"] = sum(not re.search(STEMS.get(t["c"], t["c"]), t["A"] or "", re.I) for t in hits)
    cells = {}
    for t in inj:
        cells.setdefault((t["L"], t["s"]), []).append(t)
    best = max(cells.items(), key=lambda kv: sum(t["id"] in ("correct", "related") for t in kv[1])) if cells else None
    if best:
        (bl, bs), g = best
        X.update({"bestL": bl, "bestS": str(bs).replace("0.", "."), "bestHits": sum(t["id"] in ("correct", "related") for t in g),
                  "bestN": len(g), "bestMention": sum(t["cF"] or t["cA"] for t in g)})
    fr = {f["label"]: f for f in data.get("frames", [])}
    for i, f in enumerate(data.get("frames", [])):
        X[f"frame{i}_inj"] = pct(f["injected"]["k"], f["injected"]["n"])
        X[f"frame{i}_ctl"] = pct(f["control"]["k"], f["control"]["n"])
        X[f"frame{i}_injk"] = f["injected"]["k"]
        X[f"frame{i}_ctlk"] = f["control"]["k"]
    # permission condition vs default at the same settings
    perm = data.get("permission", {}).get("trials", [])
    if perm:
        pi = [t for t in perm if t["k"] == "inject"]; pc = [t for t in perm if t["k"] == "control"]
        Ls = {t["L"] for t in pi}; Ss = {t["s"] for t in pi}
        di = [t for t in inj if t["L"] in Ls and t["s"] in Ss]
        X.update({
            "permN": len(perm), "permInj": len(pi), "permCtl": len(pc),
            "permYes": sum(t["cls"] == "yes" for t in pi), "permCtlYes": sum(t["cls"] == "yes" for t in pc),
            "permHits": sum(t["id"] in ("correct", "related") for t in pi),
            "defSameInj": len(di), "defYesSame": sum(t["cls"] == "yes" for t in di),
            "defHitsSame": sum(t["id"] in ("correct", "related") for t in di),
            "permPolicy": sum(t["pol"] for t in perm), "defPolicySame": sum(t["pol"] for t in di + ctl), "defSameN": len(di + ctl),
            "permNoCannot": pct(sum(t["cls"] in ("no", "cannot") for t in pi), len(pi)),
            "defNoCannotSame": pct(sum(t["cls"] in ("no", "cannot") for t in di), len(di)),
            "permRefuse": sum(t["cls"] == "refuse" for t in perm), "defRefuseSame": sum(t["cls"] == "refuse" for t in di + ctl),
            "permOther": sum(t["cls"] == "other" for t in pi), "defOtherSame": sum(t["cls"] == "other" for t in di),
        })
    # blind judge (Gemma 4 via Ollama)
    def jstats(ts, key, flag=None):
        g = [t for t in ts if t.get(key) and (flag is None or not t[flag])]
        k = sum(t[key] == t["c"] for t in g)
        return k, len(g)
    if any(t.get("jF") for t in low):
        for s_ in S + ["03up", "all"]:
            if s_ == "all":
                g = inj
            elif s_ == "03up":
                g = [t for t in inj if t["s"] >= 0.3]
            else:
                g = [t for t in inj if t["s"] == s_]
            key = s_ if isinstance(s_, str) else str(s_).replace("0.", "")
            for fld, flag, nm in (("jF", None, "F"), ("jA", None, "A"), ("jF", "cF", "Fimp"), ("jA", "cA", "Aimp")):
                k, n = jstats(g, fld, flag)
                X[f"j{nm}_{key}"] = k; X[f"j{nm}_{key}_n"] = n; X[f"j{nm}_{key}_pct"] = pct(k, n)
        # bias-robust baseline: how often the judge says concept c when c was NOT injected
        def base_rates(key):
            pool = [t for t in low if t.get(key)]
            tot = {}
            for c in CONCEPTS:
                others = [t for t in pool if t["c"] != c]
                tot[c] = sum(t[key] == c for t in others) / max(1, len(others))
            return tot
        def perm_exp(g, key):
            """Expected hits if the injected concepts were shuffled among these same trials:
            sum over concepts of (times the judge picked it) x (times it was injected) / n."""
            if not g:
                return 0.0
            picks = Counter(t[key] for t in g); truth = Counter(t["c"] for t in g)
            return sum(picks[c] * truth[c] for c in truth) / len(g)
        br = {key: {c: 0 for c in CONCEPTS} for key in ("jF", "jA")}   # unused: kept for reference
        exp = {}
        for key in ("jF", "jA"):
            exp[key] = []
            for s_ in S:
                g = [t for t in inj if t["s"] == s_ and t.get(key)]
                exp[key].append(perm_exp(g, key) / max(1, len(g)))
            g = [t for t in inj if t["s"] >= 0.3 and t.get(key)]
            e = perm_exp(g, key); k = sum(t[key] == t["c"] for t in g)
            X[f"{key}_03up_exp_pct"] = pct(e, len(g)); X[f"{key}_03up_excess"] = round(k - e)
            gi = [t for t in g if not t["cF" if key == "jF" else "cA"]]
            ei = perm_exp(gi, key); ki = sum(t[key] == t["c"] for t in gi)
            X[f"{key}_03up_imp_exp_pct"] = pct(ei, len(gi)); X[f"{key}_03up_imp"] = ki; X[f"{key}_03up_imp_n"] = len(gi)
            X[f"{key}_03up_imp_pct"] = pct(ki, len(gi))
        data["judge_expected"] = exp
        # the same, leaving out the judge's three habitual picks
        from collections import Counter as _C0
        fav = {w for w, _ in _C0(t["jF"] for t in low if t.get("jF")).most_common(3)}
        brF = base_rates("jF")
        def nf(ts):
            g = [t for t in ts if t.get("jF") and t["c"] not in fav]
            k = sum(t["jF"] == t["c"] for t in g); e = perm_exp(g, "jF")
            return k, len(g), e
        g3 = [t for t in inj if t["s"] >= 0.3]
        for nm, ts in (("all", g3), ("imp", [t for t in g3 if not t["cF"]]),
                       ("deny", [t for t in g3 if t["cls"] in ("no", "cannot")]),
                       ("denyimp", [t for t in g3 if t["cls"] in ("no", "cannot") and not t["cF"]]),
                       ("low", [t for t in inj if t["s"] < 0.3])):
            k, n, e = nf(ts)
            X[f"nf_{nm}"] = k; X[f"nf_{nm}_n"] = n; X[f"nf_{nm}_pct"] = pct(k, n); X[f"nf_{nm}_exp_pct"] = pct(e, n, 1)
            X[f"nf_{nm}_exp"] = round(e, 1); X[f"nf_{nm}_ratio"] = round(k / e) if e else None
        X["favWords"] = ", ".join(sorted(fav))
        perm_ = data.get("permission", {}).get("trials", [])
        if perm_:
            pi = [t for t in perm_ if t["k"] == "inject" and t.get("jF") and t["c"] not in fav]
            Ls = {t["L"] for t in perm_ if t["k"] == "inject"}; Ss = {t["s"] for t in perm_ if t["k"] == "inject"}
            di = [t for t in inj if t["L"] in Ls and t["s"] in Ss and t.get("jF") and t["c"] not in fav]
            X["permJudgeNF"] = sum(t["jF"] == t["c"] for t in pi); X["permJudgeNF_n"] = len(pi)
            X["defJudgeNF"] = sum(t["jF"] == t["c"] for t in di); X["defJudgeNF_n"] = len(di)
        from collections import Counter as _C
        cg = _C(t["jF"] for t in ctl if t.get("jF"))
        X["judgeCtlTop"] = ", ".join(f"{w} ({c})" for w, c in cg.most_common(4))
        allg = _C(t["jF"] for t in low if t.get("jF"))
        top3 = allg.most_common(3)
        X["judgeTop3"] = ", ".join(w for w, _ in top3)
        X["judgeTop3Pct"] = pct(sum(c for _, c in top3), sum(allg.values()))
        X["judgeCtlN"] = sum(cg.values())
        X["judgeChancePct"] = pct(1, len(CONCEPTS), 1)
        # how often the judge names the concept when gpt-oss explicitly denied
        den = [t for t in inj if t["cls"] in ("no", "cannot") and t.get("jF")]
        X["jDeny"] = sum(t["jF"] == t["c"] for t in den); X["jDeny_n"] = len(den); X["jDeny_pct"] = pct(X["jDeny"], len(den))
        den3 = [t for t in den if t["s"] >= 0.3]
        X["jDeny3"] = sum(t["jF"] == t["c"] for t in den3); X["jDeny3_n"] = len(den3); X["jDeny3_pct"] = pct(X["jDeny3"], len(den3))
        deni = [t for t in den if not t["cF"]]
        X["jDenyImp"] = sum(t["jF"] == t["c"] for t in deni); X["jDenyImp_n"] = len(deni); X["jDenyImp_pct"] = pct(X["jDenyImp"], len(deni))
    # medium effort vs low effort at the same settings
    med = data.get("medium", {}).get("trials", [])
    if med:
        mi = [t for t in med if t["k"] == "inject"]
        mc = [t for t in med if t["k"] == "control"]
        Ls = {t["L"] for t in mi}; Ss = {t["s"] for t in mi}
        li = [t for t in inj if t["L"] in Ls and t["s"] in Ss]
        words = lambda ts: round(sum(t["aw"] for t in ts) / max(1, len(ts)))
        X.update({
            "medN": len(med), "medInj": len(mi), "medCtl": len(mc),
            "medYes": sum(t["cls"] == "yes" for t in mi), "medCtlYes": sum(t["cls"] == "yes" for t in mc),
            "medHits": sum(t["id"] in ("correct", "related") for t in mi),
            "medNone": sum(t["cls"] == "none" for t in mi),
            "medWords": words(med), "lowWordsSame": words(li + ctl),
            "medPolicy": sum(t["pol"] for t in med), "lowPolicySame": sum(t["pol"] for t in li + ctl),
            "lowSameN": len(li + ctl),
            "medCA": sum(t["cA"] for t in mi), "lowCASame": sum(t["cA"] for t in li), "lowSameInj": len(li),
            "lowYesSame": sum(t["cls"] == "yes" for t in li), "lowHitsSame": sum(t["id"] in ("correct", "related") for t in li),
        })
    # experiment B
    ce = data.get("control_exp", [])
    if ce and ce[0].get("tF"):
        by = {}
        for r in ce:
            by.setdefault(r["c"], {})[r["think"]] = r
        def spec(L, subset=None):
            dd = []
            for c, pr in by.items():
                if True not in pr or False not in pr:
                    continue
                if subset == "nomention" and re.search(r"\b" + re.escape(c.rstrip("s")), pr[True]["A"] or "", re.I):
                    continue
                tt, ff = pr[True], pr[False]
                dd.append((tt["tF"][str(L)] - tt["oF"][str(L)]) - (ff["tF"][str(L)] - ff["oF"][str(L)]))
            return np.array(dd)
        for L in (8, 12, 16, 20, 21):
            a = spec(L)
            X[f"ctl_pos_{L}"] = int((a > 0).sum()); X[f"ctl_n_{L}"] = len(a)
            b = spec(L, "nomention")
            X[f"ctl_pos_nm_{L}"] = int((b > 0).sum()); X[f"ctl_n_nm_{L}"] = len(b)
        X["ctl_cot_mention_think"] = sum(bool(re.search(r"\b" + re.escape(r["c"].rstrip("s")), r["A"] or "", re.I)) for r in ce if r["think"])
        X["ctl_cot_mention_dont"] = sum(bool(re.search(r"\b" + re.escape(r["c"].rstrip("s")), r["A"] or "", re.I)) for r in ce if not r["think"])
        X["ctl_copied"] = sum(bool(r["copied"]) for r in ce)
        X["ctl_n"] = len(ce)
    # calibration (vector methods)
    cal = data.get("calib2", [])
    for m in ("last_token", "response_mean", "word_tokens"):
        g = [r for r in cal if r["m"] == m and r["a"] == 0.3]
        X[f"cal_{m}_03"] = f"{sum(bool(r['hit']) for r in g)} of {len(g)}"
        X[f"cal_{m}_03_coh"] = f"{sum(bool(r['coh']) for r in g)} of {len(g)}"
    vm = data.get("vectors", {})
    if vm:
        rn, vn = vm.get("resid_norm", {}), vm.get("mean_vec_norm", {})
        X["resid0"] = round(rn.get("0", 0)); X["resid23"] = round(rn.get("23", 0))
        X["residGrowth"] = round(rn.get("23", 1) / rn.get("0", 1))
        X["vecSecs"] = round(vm.get("secs", 0))
        for L in (8, 12, 16, 20):
            X[f"natural_{L}"] = round(vn.get(str(L), 0) / rn.get(str(L), 1), 2)
    return X


HARD_Q = {"Do spiders have six legs?", "Is 7 an even number?"}   # wrong even with nothing injected


def later_runs(data, J):
    """The two morning-after controls: random vectors at more settings, and unrelated yes/no questions."""
    X = data["extra_stats"]
    rows = jl("inject_random2.jsonl")
    if rows:
        tr = compact_trials(rows, "inject_random2.jsonl", J)
        data["random2"] = {"trials": tr}
        low = data["low"]["trials"]
        cells = {(t["L"], t["s"]) for t in tr}
        X["rnd2N"] = len(tr); X["rnd2Yes"] = sum(t["cls"] == "yes" for t in tr)
        X["rnd2Cells"] = len(cells)
        X["conc11Yes"] = sum(t["cls"] == "yes" for t in low if t["k"] == "inject" and (t["L"], t["s"]) in cells)
        X["rnd2Apple"] = sum(1 for t in tr if t["cls"] == "yes" and (t.get("named") or "").lower() == "apple")
        allr = tr + [t for t in low if t["k"] == "random"]
        for s_ in (0.3, 0.4, 0.6):
            key = str(s_).replace("0.", "")
            r_ = [t for t in allr if t["s"] == s_]; c_ = [t for t in low if t["k"] == "inject" and t["s"] == s_]
            X[f"rndYes{key}"] = sum(t["cls"] == "yes" for t in r_); X[f"rndN{key}"] = len(r_)
            X[f"concYes{key}"] = sum(t["cls"] == "yes" for t in c_); X[f"concN{key}"] = len(c_)
        X["rndNone6"] = sum(t["cls"] == "none" for t in allr if t["s"] == 0.6)
    # figures cited by the adversarial-review fixes
    low = data["low"]["trials"]; inj = [t for t in low if t["k"] == "inject"]
    ctl_pool = [t for run in ("low", "medium", "permission") for t in data.get(run, {}).get("trials", []) if t["k"] == "control"]
    X["nCtlPool"] = len(ctl_pool); X["yesCtlPool"] = sum(t["cls"] == "yes" for t in ctl_pool)
    perm = data.get("permission", {}).get("trials", [])
    X["permCtlPol"] = sum(t["pol"] for t in perm if t["k"] == "control"); X["permCtlN"] = sum(t["k"] == "control" for t in perm)
    X["mainCtlPol"] = sum(t["pol"] for t in low if t["k"] == "control")
    STRETCH = {"horizon", "eden", "galaxy", "discomfort or doubt", "light"}
    hits = [t for t in inj if t["id"] in ("correct", "related")]
    X["hitsStrictRelated"] = sum(t["id"] == "related" and (t.get("named") or "").lower() not in STRETCH for t in hits)
    X["hitsStretch"] = sum(t["id"] == "related" and (t.get("named") or "").lower() in STRETCH for t in hits)
    ADJ = {"secrecy", "illusions", "mirrors", "memories", "algorithms", "labyrinths", "kaleidoscopes"}
    X["adjYes"] = sum(t["cls"] == "yes" for t in inj if t["c"] in ADJ); X["adjN"] = sum(t["c"] in ADJ for t in inj)
    X["nonAdjYes"] = sum(t["cls"] == "yes" for t in inj if t["c"] not in ADJ); X["nonAdjN"] = sum(t["c"] not in ADJ for t in inj)
    if rows:
        allr = data["random2"]["trials"] + [t for t in low if t["k"] == "random"]
        for s_ in (0.3, 0.6):
            key = str(s_).replace("0.", "")
            X[f"concNone{key}"] = sum(t["cls"] == "none" for t in inj if t["s"] == s_)
            X[f"rndNoneAll{key}"] = sum(t["cls"] == "none" for t in allr if t["s"] == s_)
    yb = jl("yesbias_graded.jsonl")
    if yb:
        data["yesbias"] = [{"t": r["trial"], "k": r["kind"], "L": r["layer"], "s": r["strength"], "c": r["concept"],
                            "q": r["question"], "exp": r["expected"], "said": r["said"], "n": r["n_tokens"], "max": r["hit_max"],
                            "A": clip(r["analysis"], 900), "F": clip(r["final"], 700)} for r in yb]
        def grp(r):
            return "Ctl" if r["kind"] == "control" else ("3" if r["strength"] == 0.3 else "6")
        X["ybN"] = len(yb); X["ybCtlN"] = sum(r["kind"] == "control" for r in yb)
        for g in ("Ctl", "3", "6"):
            known = [r for r in yb if grp(r) == g and r["expected"] == "no" and r["question"] not in HARD_Q]
            X[f"ybKnown{g}Yes"] = sum(r["said"] == "yes" for r in known)
            X[f"ybKnown{g}Ans"] = sum(r["said"] is not None for r in known)
            X[f"ybKnown{g}N"] = len(known)
            yq = [r for r in yb if grp(r) == g and r["expected"] == "yes"]
            X[f"ybYesQ{g}Ans"] = sum(r["said"] is not None for r in yq); X[f"ybYesQ{g}N"] = len(yq)
            X[f"ybYesQ{g}Right"] = sum(r["said"] == "yes" for r in yq)
            hard = [r for r in yb if grp(r) == g and r["question"] in HARD_Q]
            X[f"ybHard{g}Yes"] = sum(r["said"] == "yes" for r in hard); X[f"ybHard{g}Ans"] = sum(r["said"] is not None for r in hard)
            allg = [r for r in yb if grp(r) == g]
            X[f"ybNone{g}"] = sum(r["said"] is None for r in allg); X[f"ybAll{g}N"] = len(allg)

def review_runs(data, J):
    """Job 010/011: remove-before-answer trials, structured null vectors, and the yes/no readout."""
    import numpy as np
    X = data["extra_stats"]
    low = data["low"]["trials"]
    cells = lambda t: t["L"] in (12, 16) and t["s"] in (0.3, 0.4, 0.6)
    lat = jl("latent.jsonl")
    if lat:
        tr = compact_trials(lat, "latent.jsonl", J)
        data["latent"] = {"trials": tr}
        X["latN"] = len(tr); X["latYes"] = sum(t["cls"] == "yes" for t in tr)
        X["latHits"] = sum(t["id"] in ("correct", "related") for t in tr)
        X["latNone"] = sum(t["cls"] == "none" for t in tr)
        same = [t for t in low if t["k"] == "inject" and cells(t)]
        X["onSameYes"] = sum(t["cls"] == "yes" for t in same); X["onSameN"] = len(same)
        X["onSameHits"] = sum(t["id"] in ("correct", "related") for t in same)
        X["onSameNone"] = sum(t["cls"] == "none" for t in same)
    nul = jl("null2.jsonl")
    if nul:
        tr = compact_trials(nul, "null2.jsonl", J)
        data["null2"] = {"trials": tr}
        for k, nm in (("split", "Split"), ("baseword", "Base")):
            g = [t for t in tr if t["k"] == k]
            X[f"null{nm}Yes"] = sum(t["cls"] == "yes" for t in g); X[f"null{nm}N"] = len(g)
            X[f"null{nm}Hits"] = sum(t["id"] in ("correct", "related") for t in g)
        rnd = [t for t in data.get("random2", {}).get("trials", []) + [t for t in low if t["k"] == "random"] if cells(t)]
        X["rndSameYes"] = sum(t["cls"] == "yes" for t in rnd); X["rndSameN"] = len(rnd)
    yn = jl("yn_logits.jsonl")
    if yn:
        lo = lambda r: float(np.log(r["p_yes"] + 1e-9) - np.log(r["p_no"] + 1e-9))
        ctl = {r["prompt"]: r for r in yn if r["kind"] == "control"}
        X["ynCtlYesPct"] = f"{100 * ctl['detect']['p_yes']:.0f}%"; X["ynCtlNoPct"] = f"{100 * ctl['detect']['p_no']:.0f}%"
        X["ynFactCtlYesPct"] = f"{100 * np.mean([ctl[q]['p_yes'] for q in ('q0', 'q1', 'q2')]):.1f}%"
        by = defaultdict(dict)
        for r in yn:
            if r["kind"] != "control":
                by[r["vid"]][r["prompt"]] = lo(r) - lo(ctl[r["prompt"]])
        ex = defaultdict(list)
        for vid, d in by.items():
            k, L_, s_, t_, _ = vid.split("|")
            ex[(k, t_, int(L_), float(s_))].append(d["detect"] - np.mean([d[q] for q in ("q0", "q1", "q2")]))
        rng = np.random.default_rng(0)
        def pooled(k, t_, layers=(12, 16)):
            x = np.array(sum((ex[(k, t_, L_, s_)] for L_ in layers for s_ in (0.2, 0.3, 0.4)), []))
            bs = [rng.choice(x, len(x)).mean() for _ in range(3000)]
            return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        for k, t_, nm in (("concept", "off", "ConOff"), ("concept", "on", "ConOn"), ("split", "off", "SplitOff"), ("random", "off", "RndOff")):
            m, a, b = pooled(k, t_)
            X[f"yn{nm}"] = f"{m:+.2f}"; X[f"yn{nm}Lo"] = f"{a:+.2f}"; X[f"yn{nm}Hi"] = f"{b:+.2f}"
        X["ynConOffOdds"] = f"{np.exp(pooled('concept', 'off')[0]):.1f}"
        m8 = np.mean(sum((ex[("concept", "off", 8, s_)] + ex[("split", "off", 8, s_)] for s_ in (0.2, 0.3, 0.4)), []))
        m20 = np.mean(sum((ex[("concept", "off", 20, s_)] + ex[("split", "off", 20, s_)] for s_ in (0.2, 0.3, 0.4)), []))
        X["ynL8"] = f"{m8:+.2f}"; X["ynL20"] = f"{m20:+.2f}"
        data["yn_cells"] = [{"k": k[0], "t": k[1], "L": k[2], "s": k[3], "m": float(np.mean(v)), "n": len(v)} for k, v in sorted(ex.items())]


    mt = jl("matcher.jsonl")
    if mt:
        import random as _r
        con = [m for m in mt if m["truth"]]
        main = [m for m in con if m["source"] == "inject.jsonl"]
        X["mtMainHits"] = sum(m["pick"] == m["truth"] for m in main); X["mtMainN"] = len(main)
        picks = [m["pick"] for m in main]; truth = [m["truth"] for m in main]; rr = _r.Random(0); sims = []
        for _ in range(10000):
            rr.shuffle(truth); sims.append(sum(p_ == t_ for p_, t_ in zip(picks, truth)))
        X["mtMainNull"] = f"{sum(sims) / len(sims):.1f}"; X["mtMainNullMax"] = max(sims)
        hand = [m for m in main if m["hand_id"] in ("correct", "related")]
        X["mtAgree"] = sum(m["pick"] == m["truth"] for m in hand); X["mtHandN"] = len(hand)
        lat = [m for m in con if m["source"] == "latent.jsonl"]
        X["mtLatHits"] = sum(m["pick"] == m["truth"] for m in lat); X["mtLatN"] = len(lat)

    fc = jl("forced_choice.jsonl")
    if fc:
        fctl = {(r["frame"], r["ci"], r["sh"]): r for r in fc if r["kind"] == "control"}
        def fcs(frame, kind, timing, s_):
            g = [r for r in fc if r["frame"] == frame and r["kind"] == kind and r["timing"] == timing and r["strength"] == s_]
            if not g:
                return None
            acc, rank = [], []
            for r in g:
                q = np.array(r["opt_p"]); q = q / q.sum(); t = r["target_opt"]
                acc.append(int(np.argmax(q) == t)); rank.append(int((q > q[t]).sum()) + 1)
            return 100 * np.mean(acc), np.mean(rank), len(g)
        for frame, nm in (("word", "Det"), ("nword", "Neu"), ("detect", "Num")):
            for kind, timing, kn in (("concept", "off", "Off"), ("concept", "on", "On"), ("split", "off", "Split"), ("random", "off", "Rnd")):
                for s_ in (0.2, 0.3, 0.4):
                    r_ = fcs(frame, kind, timing, s_)
                    if r_:
                        key = f"fc{nm}{kn}{str(s_).replace('0.', '')}"
                        X[key] = f"{r_[0]:.0f}%"; X[key + "Rank"] = f"{r_[1]:.1f}"
        X["fcN"] = len([r for r in fc if r["kind"] != "control"])
        # per-concept table: how well each vector is identified on the clean test
        ct = []
        for ci, cname in enumerate(CONCEPTS):
            row = {"c": cname}
            for frame, key in (("nword", "neu"), ("word", "det")):
                g = [r for r in fc if r["frame"] == frame and r["kind"] == "concept" and r["timing"] == "off" and r["ci"] == ci]
                rk, top = [], []
                for r in g:
                    q = np.array(r["opt_p"]); q = q / q.sum(); t = r["target_opt"]
                    rk.append(int((q > q[t]).sum()) + 1); top.append(int(np.argmax(q) == t))
                row[key + "Rank"] = round(float(np.mean(rk)), 1) if rk else None
                row[key + "Top"] = round(100 * float(np.mean(top))) if top else None
            low_ = [t for t in data["low"]["trials"] if t["k"] == "inject" and t["c"] == cname]
            row["claims"] = sum(t["cls"] == "yes" for t in low_); row["n"] = len(low_)
            jj = [t for t in low_ if t["s"] >= 0.3 and t.get("jF")]
            row["judge"] = sum(t["jF"] == cname for t in jj); row["judgeN"] = len(jj)
            ct.append(row)
        data["concept_table"] = sorted(ct, key=lambda r: r["neuRank"])
        X["ctGood"] = sum(r["neuRank"] <= 5 for r in ct); X["ctPoor"] = sum(r["neuRank"] >= 10 for r in ct)
        X["bellsRank"] = next(r["neuRank"] for r in ct if r["c"] == "bells")

    mm = jl("match_matrix.jsonl")
    if mm:
        lo = lambda r: float(np.log(r["p_yes"] + 1e-9) - np.log(r["p_no"] + 1e-9))
        mc = {(r["frame"], r["ask"]): lo(r) for r in mm if r["kind"] == "control"}
        rng2 = np.random.default_rng(0)
        for fr, nm in (("introspect", "Int"), ("sentence", "Sen")):
            ds, ranks = [], []
            for Lr in (12, 16):
                S = np.zeros((36, 36))
                for r in mm:
                    if r["frame"] == fr and r["kind"] != "control" and r["layer"] == Lr:
                        S[r["ci"], r["ask"]] = lo(r) - mc[(fr, r["ask"])]
                col = (S.sum(0) - np.diag(S)) / 35
                ds += list(np.diag(S) - col)
                ranks += [int((S[:, w] > S[w, w]).sum()) + 1 for w in range(36)]
            ds = np.array(ds); bs = [rng2.choice(ds, len(ds)).mean() for _ in range(3000)]
            X[f"mx{nm}"] = f"{ds.mean():+.2f}"; X[f"mx{nm}Lo"] = f"{np.percentile(bs, 2.5):+.2f}"; X[f"mx{nm}Hi"] = f"{np.percentile(bs, 97.5):+.2f}"
            X[f"mx{nm}Rank"] = f"{np.mean(ranks):.1f}"
        X["mxN"] = len(mm)
    yn2 = jl("yn2_logits.jsonl")
    if yn2:
        lo = lambda r: float(np.log(r["p_yes"] + 1e-9) - np.log(r["p_no"] + 1e-9))
        ctl = {r["prompt"]: r for r in yn2 if r["kind"] == "control"}
        grp = {r["prompt"]: r["group"] for r in yn2}
        by = defaultdict(dict)
        for r in yn2:
            if r["kind"] != "control":
                by[r["vid"]][r["prompt"]] = lo(r) - lo(ctl[r["prompt"]])
        X["yn2N"] = len(ctl); X["yn2Readouts"] = len(yn2)
        firm = {p: abs(lo(ctl[p])) for p in ctl}
        for kind, nm in (("concept", "Con"), ("split", "Split")):
            vids = [v for v in by if v.startswith(kind + "|")]
            ms = {p: float(np.mean([by[v][p] for v in vids])) for p in ctl}
            others = [p for p in ctl if grp[p] != "detect"]
            x = np.array([firm[p] for p in others]); y = np.array([ms[p] for p in others])
            b, a = np.polyfit(x, y, 1)
            sd = float(np.std(y - (a + b * x), ddof=2))
            res = [ms[p] - (a + b * firm[p]) for p in ("detect0", "detect1")]
            X[f"yn2{nm}Res"] = f"{np.mean(res):+.2f}"; X[f"yn2{nm}ResSd"] = f"{np.mean(res) / sd:+.1f}"
            X[f"yn2{nm}R"] = f"{np.corrcoef(x, y)[0, 1]:+.2f}"
            X[f"yn2{nm}Detect0"] = f"{ms['detect0']:+.2f}"
            near = sorted([p for p in ctl if firm[p] < 1.7], key=lambda p: -ms[p])
            X[f"yn2{nm}Rank"] = f"{near.index('detect0') + 1} of {len(near)}"
        X["yn2AtlanticCon"] = f"{float(np.mean([by[v]['fact_no3'] for v in by if v.startswith('concept|')])):+.2f}"


def main():
    J = load_judge()
    data = {"concepts": CONCEPTS, "stems": STEMS, "judge_chance": 1 / len(CONCEPTS)}
    for src, key in [("inject.jsonl", "low"), ("inject_medium.jsonl", "medium"),
                     ("inject_permission.jsonl", "permission")]:
        rows = jl(src)
        if not rows:
            continue
        trials = compact_trials(rows, src, J)
        data[key] = {"trials": trials, "agg": aggregate(trials), "judge_prior": judge_priors(trials)}
    data["frames"] = frames(data.get("low", {}).get("trials", []))
    data["control_exp"] = [{
        "c": r["concept"], "think": r["think"], "sent": r["sentence"],
        "A": clip(r["analysis"], 600), "F": clip(r["final"], 300), "copied": r["copied"],
        "tF": r.get("cos_final_target"), "oF": r.get("cos_final_others"),
        "tA": r.get("cos_analysis_target"), "oA": r.get("cos_analysis_others"),
    } for r in jl("control.jsonl")]
    data["showcase"] = [{"c": r["concept"], "L": r["layer"], "s": r["strength"],
                         "A": clip(r["analysis"], 600), "F": clip(r["final"], 900)} for r in jl("showcase.jsonl")]
    calib = jl("calib2.jsonl")
    data["calib2"] = [{"m": r["method"], "L": r["layer"], "a": r["alpha"], "c": r["concept"],
                       "hit": r.get("in_text"), "coh": bool(r.get("has_final")) and (r.get("distinct2") or 0) > 0.6,
                       "F": clip(r["final"] or r["text"], 400)} for r in calib]
    for p in ["smoke.json", "vectors.npz.json", "calib2_summary.json", "inject_meta.json",
              "inject_medium_meta.json", "inject_permission_meta.json"]:
        if (R / p).exists():
            d = json.load(open(R / p))
            if p == "smoke.json":
                d = {k: d[k] for k in ["env", "load_secs", "n_layers", "speed", "peak_mem_gb",
                                       "detect_prompt_tokens", "inject_start", "detect_prompt"] if k in d}
                d["speed"] = {k: v for k, v in d["speed"].items() if k != "text"}
            if p == "calib2_summary.json":
                d = {k: d[k] for k in ["agreement", "vec_norm_ratio", "tally"] if k in d}
            data[p.replace(".json", "").replace(".npz", "")] = d
    data["extra_stats"] = extra_stats(data)
    later_runs(data, J)
    review_runs(data, J)
    json.dump(data, open(OUT, "w"), separators=(",", ":"), default=str)
    print("wrote", OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
    for key in ["low", "medium", "permission"]:
        if key in data:
            print(key, len(data[key]["trials"]), "trials")


if __name__ == "__main__":
    main()
