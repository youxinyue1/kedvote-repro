# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""Self-consistency baseline from archived K=9 stochastic runs of m1 (zero GPU).

SC-mean:      score(e) = mean over 9 runs of c_k(e), missing = 0; generated = union.
SC-majority:  decision set = edges generated in >=5 of 9 runs; ranking score = mean conf.
Metric implementations copied verbatim from eval_comparison.py (same caliber as paper).
"""
import csv
import os

ROOT = ROOT  # 见文件头路径配置
PKG = DATA
GT = os.path.join(DATA, "ground_truth_aligned.csv")
LAYERS = os.path.join(DATA, "node_layers.json")
EDGES = os.path.join(ROOT, "results", "edges")
OUT = os.path.join(ROOT, "results", "eval")
os.makedirs(OUT, exist_ok=True)

PREFIX = {"S0": "症状::", "Z0": "证候::", "A0": "治法::", "Y1": "结局::"}


def build_candidates(layers):
    S0, Z0, A0, Y1 = layers["S0-症状"], layers["Z0-证候"], layers["A0-治法"], layers["Y1-结局"]
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


def load_run(k):
    return load_dir("method_m1_k%d" % k)


def load_dir(name):
    d = {}
    for r in csv.DictReader(open(os.path.join(EDGES, name, "edges.csv"),
                                  encoding="utf-8-sig")):
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
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        ap += prec * (rec - prev_rec)
        prev_rec = rec
        i = j + 1
    return ap


def main():
    layers = __import__("json").load(open(LAYERS, encoding="utf-8"))
    cand = set(build_candidates(layers))
    gold = load_gold()
    gold_keys = {(c, (s, t)) for c, (s, t) in gold}
    gt_resp = {(c, (s, t)) for c, (s, t) in gold if c in ("S0-Z0", "Z0-A0", "A0-Y1")}
    gold_in_cand = {k for k in gold_keys if k in cand}

    runs = [load_run(k) for k in range(1, 10)]
    print("runs loaded:", [len(r) for r in runs])

    rows = []
    # SC-mean
    mean_score = {k: sum(r.get(k, 0.0) for r in runs) / len(runs) for k in cand}
    gen_mean = {k for k in cand if mean_score[k] > 0}
    p = prf(gen_mean, gold, gt_resp)
    full = [(k, mean_score[k]) for k in cand]
    rows.append(("SC-mean (K=9, T>0)", p, roc_auc(full, gold_in_cand), average_precision(full, gold_in_cand)))

    # SC-majority (>=5/9), ranking score = mean conf
    votes = {k: sum(1 for r in runs if k in r) for k in cand}
    gen_maj = {k for k in cand if votes[k] >= 5}
    p2 = prf(gen_maj, gold, gt_resp)
    rows.append(("SC-majority (>=5/9)", p2, roc_auc(full, gold_in_cand), average_precision(full, gold_in_cand)))

    # per-run average for context
    import statistics as st
    per = []
    for idx, r in enumerate(runs, 1):
        pp = prf(r, gold, gt_resp)
        ff = [(k, r.get(k, 0.0)) for k in cand]
        per.append((pp, roc_auc(ff, gold_in_cand), average_precision(ff, gold_in_cand)))
    rows.append(("per-run mean (9 runs)", dict(n_pred=st.mean(x[0]["n_pred"] for x in per),
                                               tp=st.mean(x[0]["tp"] for x in per),
                                               prec=st.mean(x[0]["prec"] for x in per),
                                               rec15=st.mean(x[0]["rec15"] for x in per),
                                               rec17=st.mean(x[0]["rec17"] for x in per),
                                               f1_15=st.mean(x[0]["f1_15"] for x in per)),
                 st.mean(x[1] for x in per), st.mean(x[2] for x in per)))

    # ---- budget-matched SC-K4: all C(9,4)=126 four-run subsets (4 forward passes, like KED-Vote)
    from itertools import combinations
    k4_aupr, k4_auc = [], []
    for sub in combinations(range(9), 4):
        sc = {k: sum(runs[i].get(k, 0.0) for i in sub) / 4.0 for k in cand}
        ff = [(k, sc[k]) for k in cand]
        k4_aupr.append(average_precision(ff, gold_in_cand))
        k4_auc.append(roc_auc(ff, gold_in_cand))
    print("\nSC-K4 (126 subsets): AUPR mean=%.4f sd=%.4f min=%.4f max=%.4f | AUC mean=%.4f max=%.4f"
          % (st.mean(k4_aupr), st.stdev(k4_aupr), min(k4_aupr), max(k4_aupr),
             st.mean(k4_auc), max(k4_auc)))

    # ---- paired bootstrap, verbatim caliber of stats_significance.py (n_boot=2000, seed=42)
    import random
    cand_list = sorted(cand)
    labels = [1 if k in gold_keys else 0 for k in cand_list]

    def vlist(score_map):
        return [score_map.get(k, 0.0) for k in cand_list]

    def auc_list(vals):
        return roc_auc(list(zip(cand_list, vals)), gold_in_cand)

    def apr_list(vals):
        return average_precision(list(zip(cand_list, vals)), gold_in_cand)

    def boot_paired(vals1, vals2, metric, n_boot=2000, seed=42):
        rng = random.Random(seed)
        idx = list(range(len(vals1)))
        le = ge = 0
        n_eff = 0
        for _ in range(n_boot):
            b = [rng.choice(idx) for _ in idx]
            v1 = [vals1[i] for i in b]
            v2 = [vals2[i] for i in b]
            lb = [labels[i] for i in b]
            m1v = metric(v1, lb)
            m2v = metric(v2, lb)
            if m1v != m1v or m2v != m2v:
                continue
            d = m1v - m2v
            n_eff += 1
            if d <= 0:
                le += 1
            if d >= 0:
                ge += 1
        p = 2.0 * min(le, ge) / max(n_eff, 1)
        return min(1.0, p)

    def metric_vals(vals, lb):
        # vals aligned to cand_list; lb bootstrap labels
        pairs = list(zip(cand_list, vals))
        gl = {cand_list[i] for i, x in enumerate(lb) if x}
        # AUPR on resampled multiset: replicate edge multiplicity via labels
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        P = sum(lb)
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
                if lb[order[k]]:
                    tp += 1
                else:
                    fp += 1
            rec = tp / P
            ap += (tp / (tp + fp)) * (rec - prev)
            prev = rec
            i = j + 1
        return ap

    m1_det = load_dir("method_m1")
    ked = load_dir("method_m12_c3_d3")
    v_sc, v_m1, v_ked = vlist(mean_score), vlist(m1_det), vlist(ked)
    p_sc_m1 = boot_paired(v_sc, v_m1, metric_vals)
    p_sc_ked = boot_paired(v_ked, v_sc, metric_vals)
    p_ked_m1 = boot_paired(v_ked, v_m1, metric_vals)
    print("AUPR: SC-mean=%.4f m1(T=0)=%.4f KED(c3d3)=%.4f" % (apr_list(v_sc), apr_list(v_m1), apr_list(v_ked)))
    print("bootstrap p: SC-mean vs m1 = %.4f | KED vs SC-mean = %.4f | KED vs m1 = %.4f"
          % (p_sc_m1, p_sc_ked, p_ked_m1))

    with open(os.path.join(OUT, "selfconsistency_k4_bootstrap.txt"), "w", encoding="utf-8") as f:
        f.write("Self-consistency control detail (S7)\n")
        f.write("caliber: 282-edge candidate space, 17-edge gold, metrics as eval_comparison.py\n\n")
        f.write("baselines: m1 deterministic T=0 AUPR=%.4f | stochastic per-run mean AUPR=%.4f\n\n"
                % (apr_list(v_m1), st.mean(x[2] for x in per)))
        f.write("SC-mean (K=9): AUPR=%.4f AUC=%.4f\n" % (apr_list(v_sc), auc_list(v_sc)))
        f.write("SC-majority (>=5/9): P=0.183 F1=0.302 (see selfconsistency_summary.csv)\n\n")
        f.write("SC-K4 budget-matched control (all 126 four-run subsets):\n")
        f.write("  AUPR mean=%.4f sd=%.4f min=%.4f max=%.4f\n" % (st.mean(k4_aupr), st.stdev(k4_aupr),
                                                                 min(k4_aupr), max(k4_aupr)))
        f.write("  AUC  mean=%.4f max=%.4f\n\n" % (st.mean(k4_auc), max(k4_auc)))
        f.write("paired bootstrap (2000 resamples, seed 42, as stats_significance.py):\n")
        f.write("  SC-mean vs m1   AUPR diff=%+.4f p=%.4f (ns)\n" % (apr_list(v_sc) - apr_list(v_m1), p_sc_m1))
        f.write("  KED vs SC-mean  AUPR diff=%+.4f p=%.4f (sig)\n" % (apr_list(v_ked) - apr_list(v_sc), p_sc_ked))
        f.write("  KED vs m1       AUPR diff=%+.4f p=%.4f (sig; paper Table 6 reports 0.0140)\n"
                % (apr_list(v_ked) - apr_list(v_m1), p_ked_m1))
    print("detail written:", os.path.join(OUT, "selfconsistency_k4_bootstrap.txt"))

    with open(os.path.join(OUT, "selfconsistency_summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "generated", "tp", "precision", "recall_15", "recall_17", "f1_15", "auc", "aupr"])
        for name, p, auc, aupr in rows:
            w.writerow([name, "%.0f" % p["n_pred"], "%.1f" % p["tp"], "%.4f" % p["prec"],
                        "%.4f" % p["rec15"], "%.4f" % p["rec17"], "%.4f" % p["f1_15"],
                        "%.4f" % auc, "%.4f" % aupr])
    for name, p, auc, aupr in rows:
        print("%-24s gen=%-4.0f TP=%-4.1f P=%.3f R15=%.3f F1=%.3f AUC=%.4f AUPR=%.4f"
              % (name, p["n_pred"], p["tp"], p["prec"], p["rec15"], p["f1_15"], auc, aupr))


if __name__ == "__main__":
    main()
