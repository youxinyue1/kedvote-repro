# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""统计显著性检验（v3.3 §3.7）：DeLong AUC 配对检验 + bootstrap CI/配对差 + McNemar@top50
被检方法：KED-Vote 主配置 m12_c3_d3 与 m12_c3_d1 vs 最强单体 m1 / 旧投票 m6 / 次强单体 m9
"""
import csv
import os
import math
import random

EXP = EXP  # 见文件头路径配置
PKG = DATA
GT = os.path.join(DATA, "ground_truth_aligned.csv")
LAYERS = os.path.join(DATA, "node_layers.json")
import json

PREFIX = {"S0": "症状::", "Z0": "证候::", "A0": "治法::", "Y1": "结局::"}


def build_candidates():
    layers = json.load(open(LAYERS, encoding="utf-8"))
    S0, Z0, A0, Y1 = layers["S0-症状"], layers["Z0-证候"], layers["A0-治法"], layers["Y1-结局"]
    combos = [("S0-Z0", S0, Z0), ("S0-Y1", S0, Y1), ("Z0-A0", Z0, A0),
              ("Z0-Y1", Z0, Y1), ("A0-Y1", A0, Y1)]
    return [(c, (s, t)) for c, ls, lt in combos for s in ls for t in lt]


def load_gold():
    out = set()
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
        out.add((c, (s, t)))
    return out


def load_pred(method):
    d = {}
    csvp = os.path.join(EXP, "method_%s" % method, "edges.csv")
    if not os.path.exists(csvp):
        return d
    for r in csv.DictReader(open(csvp, encoding="utf-8-sig")):
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


def full_scores(pred, cand):
    return [pred.get(k, 0.0) for k in cand]


def auc_rank(vals, labels):
    """秩法 AUC（同分平均秩）。vals/labels 等长列表。"""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [ranks[i] for i in range(len(vals)) if labels[i]]
    neg = [ranks[i] for i in range(len(vals)) if not labels[i]]
    if not pos or not neg:
        return float("nan")
    return (sum(pos) / len(pos) - (len(pos) + 1) / 2.0) / len(neg)


def average_precision(vals, labels):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    P = sum(labels)
    if P == 0:
        return float("nan")
    tp = fp = 0
    ap = 0.0
    prev = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if labels[order[k]]:
                tp += 1
            else:
                fp += 1
        rec = tp / P
        ap += (tp / (tp + fp)) * (rec - prev)
        prev = rec
        i = j + 1
    return ap


# ---------------- DeLong ----------------
def delong_variance_components(vals, labels):
    """返回 (auc, v10, v01)：结构成分，用于配对 DeLong 方差。"""
    pos = [v for v, l in zip(vals, labels) if l]
    neg = [v for v, l in zip(vals, labels) if not l]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan"), [], []
    # V10 (positive 结构) 与 V01 (negative 结构)
    v10 = []
    for x in pos:
        v10.append(sum(1.0 if x > y else (0.5 if x == y else 0.0) for y in neg) / n0)
    v01 = []
    for y in neg:
        v01.append(sum(1.0 if x > y else (0.5 if x == y else 0.0) for x in pos) / n1)
    auc = sum(v10) / n1
    return auc, v10, v01


def delong_test(vals1, vals2, labels):
    """配对 DeLong（同一样本两评分）：返回 (auc1, auc2, z, p)。"""
    a1, v10_1, v01_1 = delong_variance_components(vals1, labels)
    a2, v10_2, v01_2 = delong_variance_components(vals2, labels)
    n1, n0 = len(v10_1), len(v01_1)
    # 协方差成分
    m1v = sum(v10_1) / n1
    m0v = sum(v01_1) / n0
    m1w = sum(v10_2) / n1
    m0w = sum(v01_2) / n0
    s10 = sum((a - m1v) * (b - m1w) for a, b in zip(v10_1, v10_2)) / (n1 - 1)
    s01 = sum((a - m0v) * (b - m0w) for a, b in zip(v01_1, v01_2)) / (n0 - 1)
    var = s10 / n1 + s01 / n0
    diff = a1 - a2
    if var <= 0:
        return a1, a2, float("nan"), float("nan")
    z = diff / math.sqrt(var)
    # 双侧 p（正态近似）
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return a1, a2, z, p


# ---------------- bootstrap ----------------
def bootstrap_paired(vals1, vals2, labels, metric, n_boot=2000, seed=42):
    """配对 bootstrap（标准双侧，含 (1+count)/(1+B) 小样本校正）：
    p = 2·(1+min(count_≤0, count_≥0)) / (1+B)，截断 1。
    校正防止 p 严格为 0（North, Cureton & Aiken 2002; Davison & Hinkley 1997）。
    分母严格用 n_boot 而非 n_eff：有 NaN 跳过迭代时仍保持 (1+count)/(1+B) 形式。
    count 仅统计非 NaN 迭代。"""
    rng = random.Random(seed)
    idx = list(range(len(vals1)))
    d0 = metric(vals1, labels) - metric(vals2, labels)
    le = ge = 0
    n_eff = 0
    for _ in range(n_boot):
        b = [rng.choice(idx) for _ in idx]
        v1 = [vals1[i] for i in b]
        v2 = [vals2[i] for i in b]
        lb = [labels[i] for i in b]
        m1 = metric(v1, lb)
        m2 = metric(v2, lb)
        if m1 != m1 or m2 != m2:
            continue
        d = m1 - m2
        n_eff += 1
        if d <= 0:
            le += 1
        if d >= 0:
            ge += 1
    # (1+count)/(1+n_boot) 小样本校正后再乘 2 取 min(1, ...)
    p = min(1.0, 2.0 * (1 + min(le, ge)) / (1 + n_boot))
    return p, d0


def mcnemar_topk(pred1, pred2, cand, gold, k=50):
    """两方法 top-k 候选边的命中/未命中配对表 → McNemar 精确检验（二项）."""
    s1 = sorted(cand, key=lambda kk: -pred1.get(kk, 0.0))[:k]
    s2 = sorted(cand, key=lambda kk: -pred2.get(kk, 0.0))[:k]
    hit1 = {kk for kk in s1 if kk in gold}
    hit2 = {kk for kk in s2 if kk in gold}
    b = len(hit1 - hit2)   # 仅 A 命中
    c = len(hit2 - hit1)   # 仅 B 命中
    n = b + c
    if n == 0:
        return b, c, 1.0
    # 精确二项双侧 p：p = 2·P(X <= min(b,c))，X~Bin(n, 0.5)（标准 McNemar 精确检验）
    def binom_p(n, k):
        from math import comb
        return comb(n, k) * 0.5 ** n
    m = min(b, c)
    p = sum(binom_p(n, i) for i in range(0, m + 1))
    return b, c, min(1.0, 2.0 * p)


def main():
    cand = build_candidates()
    gold = load_gold()
    labels = [1 if k in gold else 0 for k in cand]
    print("候选=%d 金标准=%d" % (len(cand), len(gold)))

    focus = ["m1", "m9", "m6", "m12_c3_d3", "m12_c3_d1"]
    preds = {m: load_pred(m) for m in focus}
    vals = {m: full_scores(preds[m], cand) for m in focus}
    aucs = {m: auc_rank(vals[m], labels) for m in focus}
    aprs = {m: average_precision(vals[m], labels) for m in focus}
    print("\n== 基础指标 ==")
    for m in focus:
        print("  %-12s AUC=%.4f AUPR=%.4f" % (m, aucs[m], aprs[m]))

    print("\n== DeLong 配对检验（AUC）==")
    pairs = [("m12_c3_d3", "m1"), ("m12_c3_d3", "m9"), ("m12_c3_d3", "m6"),
             ("m12_c3_d1", "m1"), ("m12_c3_d1", "m6")]
    for a, b in pairs:
        a1, a2, z, p = delong_test(vals[a], vals[b], labels)
        print("  %s(AUC=%.4f) vs %s(AUC=%.4f): z=%+.3f p=%.4f %s"
              % (a, a1, b, a2, z, p, "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))))

    print("\n== 配对 bootstrap（2000 次）==")
    for a, b in pairs:
        for name, fn in [("AUC", auc_rank), ("AUPR", average_precision)]:
            p, d0 = bootstrap_paired(vals[a], vals[b], labels, fn)
            print("  %s-%s %s 差=%.4f p=%.4f %s"
                  % (a, b, name, d0, p, "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))))

    print("\n== McNemar @top50 边命中 ==")
    for a, b in [("m12_c3_d3", "m1"), ("m12_c3_d1", "m1"), ("m12_c3_d3", "m6"), ("m12_c3_d1", "m6")]:
        bb, cc, p = mcnemar_topk(preds[a], preds[b], cand, gold, k=50)
        print("  %s vs %s: 仅A命中=%d 仅B命中=%d p=%.4f %s"
              % (a, b, bb, cc, p, "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))))

    # 落盘
    out = os.path.join(EXP, "eval", "significance_tests.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("统计显著性检验（DeLong / bootstrap / McNemar）v3.3\n")
        f.write("候选=%d 金标准=%d\n\n" % (len(cand), len(gold)))
        for m in focus:
            f.write("%-12s AUC=%.4f AUPR=%.4f\n" % (m, aucs[m], aprs[m]))
        f.write("\n")
        for a, b in pairs:
            a1, a2, z, p = delong_test(vals[a], vals[b], labels)
            f.write("DeLong  %s vs %s: z=%+.3f p=%.4f\n" % (a, b, z, p))
        f.write("\n")
        for a, b in pairs:
            for name, fn in [("AUC", auc_rank), ("AUPR", average_precision)]:
                p, d0 = bootstrap_paired(vals[a], vals[b], labels, fn)
                f.write("boot   %s-%s %s diff=%.4f p=%.4f\n" % (a, b, name, d0, p))
        f.write("\n")
        for a, b in [("m12_c3_d3", "m1"), ("m12_c3_d1", "m1"), ("m12_c3_d3", "m6"), ("m12_c3_d1", "m6")]:
            bb, cc, p = mcnemar_topk(preds[a], preds[b], cand, gold, k=50)
            f.write("McNemar@50 %s vs %s: onlyA=%d onlyB=%d p=%.4f\n" % (a, b, bb, cc, p))
    print("\n落盘: %s" % out)


if __name__ == "__main__":
    main()
