# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""Panel error-dependence and aggregation-baseline analysis (zero GPU, §4.3 inputs).

1. Spearman score-correlation matrix across the single models over the 282-edge
   candidate space (missing = 0), with within-family vs cross-family means.
2. Max / Median aggregation baselines on the primary c3 panel (m1, m9, m10, m4),
   same caliber as ked_vote.py d3 (union, missing = 0).
Metric implementations copied verbatim from eval_comparison.py (same as paper).
"""
import csv
import os
import json
import statistics as st

ROOT = ROOT  # 见文件头路径配置
PKG = DATA
GT = os.path.join(DATA, "ground_truth_aligned.csv")
LAYERS = os.path.join(DATA, "node_layers.json")
EDGES = os.path.join(ROOT, "results", "edges")
OUT = os.path.join(ROOT, "results", "eval")
os.makedirs(OUT, exist_ok=True)

PREFIX = {"S0": "症状::", "Z0": "证候::", "A0": "治法::", "Y1": "结局::"}

MODELS = ["m1", "m2", "m3", "m9", "m9b", "m9c", "m10", "m11", "m8b", "m4", "m4b"]
LABEL = {"m1": "Qwen2.5-14B", "m2": "Qwen2.5-7B", "m3": "Qwen2.5-3B",
         "m9": "R1-D-14B", "m9b": "R1-D-7B", "m9c": "R1-D-1.5B",
         "m10": "GLM-4-9B", "m11": "Yi-1.5-34B", "m8b": "Bai2-7B",
         "m4": "BianCang-14B", "m4b": "Shizhen-7B"}
FAMILY = {"m1": "Qwen", "m2": "Qwen", "m3": "Qwen",
          "m9": "R1", "m9b": "R1", "m9c": "R1",
          "m10": "GLM", "m11": "Yi", "m8b": "Bai",
          "m4": "Domain", "m4b": "Domain"}


def build_candidates(layers):
    S0 = layers["S0-症状"]
    Z0 = layers["Z0-证候"]
    A0 = layers["A0-治法"]
    Y1 = layers["Y1-结局"]
    combos = [("S0-Z0", S0, Z0), ("S0-Y1", S0, Y1), ("Z0-A0", Z0, A0),
              ("Z0-Y1", Z0, Y1), ("A0-Y1", A0, Y1)]
    return [(c, (s, t)) for c, ls, lt in combos for s in ls for t in lt]


def load_gold():
    out = []
    for r in csv.DictReader(open(GT, encoding="utf-8-sig")):
        s, t = r["source"].strip(), r["target"].strip()
        if s.startswith("症状::") and t.startswith("证候::"):
            c = "S0-Z0"
        elif s.startswith("症状::") and t.startswith("结局::"):
            c = "S0-Y1"
        elif s.startswith("证候::") and t.startswith("治法::"):
            c = "Z0-A0"
        elif s.startswith("证候::") and t.startswith("结局::"):
            c = "Z0-Y1"
        elif s.startswith("治法::") and t.startswith("结局::"):
            c = "A0-Y1"
        else:
            continue
        out.append((c, (s, t)))
    return out


def load_dir(name):
    d = {}
    p = os.path.join(EDGES, name, "edges.csv")
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        combo = r["combo"].strip()
        s, t = r["source"].strip(), r["target"].strip()
        try:
            conf = float(r.get("confidence") or "nan")
        except ValueError:
            conf = float("nan")
        if conf != conf or conf <= 0:
            continue
        if combo == "S0-Z0":
            s, t = t, s
        if combo in ("S0-Z0", "Z0-A0", "A0-Y1"):
            a, b = combo.split("-")
            key = (combo, (PREFIX[a] + s, PREFIX[b] + t))
            if key not in d or conf > d[key]:
                d[key] = conf
    return d


def prf(pred, gold, gt_resp_combo):
    all_tp = all_fp = 0
    for combo in ["S0-Z0", "Z0-A0", "A0-Y1", "S0-Y1", "Z0-Y1"]:
        g = {(s, t) for c, (s, t) in gold if c == combo}
        p = {(s, t) for c, (s, t) in pred if c == combo}
        all_tp += len(g & p)
        all_fp += len(p - g)
    prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    rec17 = all_tp / len(gold) if gold else 0.0
    rec15 = all_tp / len(gt_resp_combo) if gt_resp_combo else 0.0
    f1_15 = 2 * prec * rec15 / (prec + rec15) if (prec + rec15) else 0.0
    return dict(n_pred=len(pred), tp=all_tp, prec=prec, rec17=rec17, rec15=rec15, f1_15=f1_15)


def roc_auc(scores, gold_set):
    order = sorted(scores, key=lambda x: x[1])
    n = len(order)
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and order[j + 1][1] == order[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[k] = avg
        i = j + 1
    P = len(gold_set)
    N = n - P
    if P == 0 or N == 0:
        return float("nan")
    s = 0.0
    for k, (key, _) in enumerate(order):
        if key in gold_set:
            s += rank[k]
    return (s / P - (P + 1.0) / 2.0) / N


def average_precision(scores, gold_set):
    order = sorted(scores, key=lambda x: -x[1])
    P = len(gold_set)
    if P == 0:
        return float("nan")
    tp = fp = 0
    ap = 0.0
    prev_rec = 0.0
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and order[j + 1][1] == order[i][1]:
            j += 1
        for k in range(i, j + 1):
            if order[k][0] in gold_set:
                tp += 1
            else:
                fp += 1
        rec = tp / P
        ap += (tp / (tp + fp)) * (rec - prev_rec)
        prev_rec = rec
        i = j + 1
    return ap


def spearman(x, y):
    """Spearman rho via Pearson on ranks (ties averaged)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    layers = json.load(open(LAYERS, encoding="utf-8"))
    cand = build_candidates(layers)
    gold = load_gold()
    gold_keys = {(c, k) for c, k in gold}
    gt_resp_combo = [(c, k) for c, k in gold if c in ("S0-Z0", "Z0-A0", "A0-Y1")]

    # ---------- 1. Spearman correlation matrix ----------
    vecs = {}
    for m in MODELS:
        d = load_dir("method_" + m)
        if d is None:
            print("skip (no local edges):", m)
            continue
        vecs[m] = [d.get(k, 0.0) for k in cand]

    names = list(vecs)
    n = len(names)
    print("models included:", len(names))
    rho = {}
    for i in range(n):
        for j in range(i + 1, n):
            rho[(names[i], names[j])] = spearman(vecs[names[i]], vecs[names[j]])

    within, cross, panel_pairs = [], [], []
    for (a, b), v in rho.items():
        if FAMILY[a] == FAMILY[b]:
            within.append(v)
        else:
            cross.append(v)
    panel = ["m1", "m9", "m10", "m4"]
    for i in range(4):
        for j in range(i + 1, 4):
            key = tuple(sorted((panel[i], panel[j])))
            v = rho.get(key) or rho.get((key[1], key[0]))
            panel_pairs.append((LABEL[key[0]], LABEL[key[1]], v))

    print("within-family mean rho = %.3f (n=%d, range %.3f-%.3f)"
          % (st.mean(within), len(within), min(within), max(within)))
    print("cross-family  mean rho = %.3f (n=%d, range %.3f-%.3f)"
          % (st.mean(cross), len(cross), min(cross), max(cross)))
    print("primary panel pairwise:")
    for a, b, v in panel_pairs:
        print("   %-14s x %-14s rho=%.3f" % (a, b, v))

    # ---------- 2. Max / Median aggregation on c3 ----------
    sets = {m: load_dir("method_" + m) for m in panel}
    keys = set()
    for s in sets.values():
        keys |= set(s)
    rows = {}
    for strat in ("max", "median"):
        out = {}
        for k in keys:
            cs = [sets[m].get(k, 0.0) for m in panel]
            out[k] = max(cs) if strat == "max" else (sorted(cs)[1] + sorted(cs)[2]) / 2.0
        rows[strat] = out
    # mean reference re-computed for sanity
    mean_out = {k: sum(sets[m].get(k, 0.0) for m in panel) / 4.0 for k in keys}
    rows["mean"] = mean_out

    res = {}
    for strat, out in rows.items():
        ff = [(k, out.get(k, 0.0)) for k in cand]  # full 282-candidate caliber, missing = 0
        r = prf(set(out), gold, gt_resp_combo)
        r["auc"] = roc_auc(ff, gold_keys)
        r["aupr"] = average_precision(ff, gold_keys)
        res[strat] = r
        print("%-7s gen=%3d TP=%2d P=%.3f R15=%.3f F1=%.3f AUC=%.4f AUPR=%.4f"
              % (strat, r["n_pred"], r["tp"], r["prec"], r["rec15"], r["f1_15"], r["auc"], r["aupr"]))

    # ---------- archive ----------
    with open(os.path.join(OUT, "panel_correlation.txt"), "w", encoding="utf-8") as f:
        f.write("Panel error-dependence and aggregation baselines (§4.3 inputs)\n")
        f.write("caliber: 282-edge candidate space, 17-edge gold; Spearman over full vectors (missing=0)\n\n")
        f.write("Spearman rho matrix (%d models):\n" % n)
        f.write("        " + "".join("%9s" % LABEL[m][:8] for m in names) + "\n")
        for mi in names:
            row = []
            for mj in names:
                if mi == mj:
                    row.append("    1.00")
                else:
                    v = rho.get((mi, mj)) or rho.get((mj, mi))
                    row.append("%9.3f" % v)
            f.write("%8s" % LABEL[mi][:8] + "".join(row) + "\n")
        f.write("\nwithin-family mean rho = %.3f (n=%d)\ncross-family  mean rho = %.3f (n=%d)\n\n"
                % (st.mean(within), len(within), st.mean(cross), len(cross)))
        f.write("primary panel (c3) pairwise:\n")
        for a, b, v in panel_pairs:
            f.write("  %-14s x %-14s rho=%.3f\n" % (a, b, v))
        f.write("\naggregation baselines on c3 (union, missing=0):\n")
        for strat, r in res.items():
            f.write("  %-7s gen=%d TP=%d P=%.4f R15=%.4f F1=%.4f AUC=%.4f AUPR=%.4f\n"
                    % (strat, r["n_pred"], r["tp"], r["prec"], r["rec15"], r["f1_15"], r["auc"], r["aupr"]))
        f.write("\nnote: m8 (Baichuan2-13B) excluded — degenerate run with zero scored edges.\n")
    print("written:", os.path.join(OUT, "panel_correlation.txt"))


if __name__ == "__main__":
    main()
