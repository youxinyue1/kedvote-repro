# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""K=9 重复解码稳健性实验 · 统计（本地运行）
================================================
m1 (qwen2.5:14b, V0 提示, temperature=0.7) 独立跑批 9 次（method_m1_k1..k9），
以统一评估口径（候选空间 282 对 / 金标准 17 条）计算每次的指标，
报告 均值±标准差 / 极差，与 temperature=0 的主实验 m1 对照。

产物: eval/k9_variance.csv + eval/k9_variance.txt
"""
import os
import csv
import json
import statistics as st

import eval_comparison as ec

EXP = ec.EXP
EVAL = ec.EVAL


def eval_run(method):
    layers = json.load(open(ec.LAYERS, encoding="utf-8"))
    cand = ec.build_candidates(layers)
    gold = ec.load_gold()
    gold_keys = {(c, (s, t)) for c, (s, t) in gold}
    gt_resp = {(c, (s, t)) for c, (s, t) in gold if c in ("S0-Z0", "Z0-A0", "A0-Y1")}
    pred = ec.load_pred(method)
    p = ec.prf(pred, gold, cand, gt_resp)
    full = [(k, pred.get(k, 0.0)) for k in set(cand)]
    gk = gold_keys & set(cand)
    auc = ec.roc_auc(full, gk)
    aupr = ec.average_precision(full, gk)
    return dict(method=method, generated=len(pred), tp=p["tp"],
                prec=p["prec"], rec15=p["rec15"], f1_15=p["f1_15"],
                auc=auc, aupr=aupr, pred=pred)


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    runs = [eval_run("m1_k%d" % k) for k in range(1, 10)]
    base = eval_run("m1")   # temperature=0 主实验对照

    cols = ["generated", "tp", "prec", "rec15", "f1_15", "auc", "aupr"]
    names = {"generated": "生成边数", "tp": "TP", "prec": "P", "rec15": "R15",
             "f1_15": "F1_15", "auc": "AUC", "aupr": "AUPR"}

    lines = ["# K=9 重复解码稳健性（m1 = qwen2.5:14b V0, temperature=0.7）\n"]
    lines.append("口径与主实验一致：候选空间 282 对 / 金标准 17 条（candidate-set）。\n")
    lines.append("\n## 每次跑批指标\n")
    lines.append("| 轮次 | 生成边 | TP | P | R15 | F1 | AUC | AUPR |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        lines.append("| k=%d | %d | %d | %.3f | %.3f | %.3f | %.4f | %.4f |"
                     % (runs.index(r) + 1, r["generated"], r["tp"], r["prec"],
                        r["rec15"], r["f1_15"], r["auc"], r["aupr"]))
    lines.append("| **m1 主实验(T=0)** | %d | %d | %.3f | %.3f | %.3f | **%.4f** | **%.4f** |"
                 % (base["generated"], base["tp"], base["prec"], base["rec15"],
                    base["f1_15"], base["auc"], base["aupr"]))

    lines.append("\n## 均值 ± 标准差（k=1..9）\n")
    lines.append("| 指标 | 均值 | 标准差 | 极差 | T=0 主实验 |")
    lines.append("|---|---|---|---|---|")
    for c in cols:
        vals = [r[c] for r in runs]
        mu, sd = st.mean(vals), st.stdev(vals)
        rng = max(vals) - min(vals)
        bv = base[c]
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f |" % (names[c], mu, sd, rng, bv))

    # 边级稳定性：9 轮中出现次数 >=6 的边 / 金标准边出现频次
    from collections import Counter
    cnt = Counter()
    for r in runs:
        for k in r["pred"]:
            cnt[k] += 1
    gold_keys = {(c, (s, t)) for c, (s, t) in ec.load_gold()} & set(cnt) | \
                {(c, (s, t)) for c, (s, t) in ec.load_gold()}
    stable = {k: v for k, v in cnt.items() if v >= 6}
    lines.append("\n## 边级稳定性\n")
    lines.append("- 9 轮中每轮都出现的边（k=9）：**%d 条**；出现 ≥6 轮：%d 条；"
                 "9 轮并集：%d 条（T=0 主实验 %d 条）\n"
                 % (sum(1 for v in cnt.values() if v == 9), len(stable),
                    len(cnt), base["generated"]))
    gold_hits = Counter({k: cnt.get(k, 0) for k in gold_keys})
    always_hit = [k for k, v in gold_hits.items() if v == 9]
    lines.append("- 金标准 15 条理论边中，9 轮全命中 %d 条，≥6 轮命中 %d 条，"
                 "从未命中 %d 条\n"
                 % (len(always_hit), sum(1 for v in gold_hits.values() if v >= 6),
                    sum(1 for v in gold_hits.values() if v == 0)))

    out_md = os.path.join(EVAL, "k9_variance.txt")
    open(out_md, "w", encoding="utf-8").write("\n".join(lines))
    with open(os.path.join(EVAL, "k9_variance.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "temperature", "generated", "tp", "prec", "rec15",
                    "f1_15", "auc", "aupr"])
        for i, r in enumerate(runs, 1):
            w.writerow(["m1_k%d" % i, 0.7, r["generated"], r["tp"],
                        "%.4f" % r["prec"], "%.4f" % r["rec15"], "%.4f" % r["f1_15"],
                        "%.4f" % r["auc"], "%.4f" % r["aupr"]])
        w.writerow(["m1", 0.0, base["generated"], base["tp"],
                    "%.4f" % base["prec"], "%.4f" % base["rec15"], "%.4f" % base["f1_15"],
                    "%.4f" % base["auc"], "%.4f" % base["aupr"]])

    print("\n".join(lines))
    print("\n已写:", out_md)


if __name__ == "__main__":
    main()
