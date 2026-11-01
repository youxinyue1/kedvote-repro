# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""Leave-one-edge-out (LOEO) 敏感性分析（本地运行）
====================================================
目的：检验主要结论是否由某一条金标准边驱动（审稿人常见质疑：
"去掉某条边后结论还成立吗？"）。

方法：对 17 条金标准边逐条留一（该边改判为负例），在剩余 16 条正例上
重算各方法 AUC / AUPR（口径与 eval_comparison.py 完全一致：平均秩 AUC、
块积分 AP、候选空间 282 对、未生成边 conf=0）。

产物（写到 results/）：
  loeo_sensitivity.csv   逐折明细：left_out_edge × method × 指标
  loeo_summary.txt       汇总：基线复现 + 稳定性统计 + 结论保持计数
"""
import csv
import os

ROOT = ROOT  # 见文件头路径配置（KED_ROOT 可覆盖）
WIDE = os.path.join(ROOT, "results", "per_edge_predictions_wide.csv")
OUT_CSV = os.path.join(ROOT, "results", "loeo_sensitivity.csv")
OUT_TXT = os.path.join(ROOT, "results", "loeo_summary.txt")

M1 = "m1(Qwen14B 最强单体)"
M9 = "m9(R1-14B 次强单体)"
M4 = "m4(扁仓14B 领域)"
M5 = "m5(V2 提示增强)"
M6 = "m6(旧交集投票)"
KED = "m12_c3_d3(KED-均值 主配置)"
KED1 = "m12_c3_d1(KED-交集)"
PC = "m7(PC-D1 数据驱动)"

# 关键方法（论文主线涉及）：显示名 -> wide 表列名
METHODS = [
    (M1, "Qwen2.5-14B"),
    (M9, "R1-14B"),
    (M4, "扁仓14B"),
    (M5, "V2完整"),
    (M6, "旧交集票"),
    (KED, "KED-均值(主)"),
    (KED1, "KED-交集"),
    (PC, "PC-D1"),
]

# 结论判据（每折重算后检查）
def conclusions(metrics):
    ked = metrics[KED]
    m1 = metrics[M1]
    pc = metrics[PC]
    best_single_aupr = max(metrics[k]["aupr"] for k in (M1, M9, M4, M5))
    best_single_auc = max(metrics[k]["auc"] for k in (M1, M9, M4, M5))
    return {
        "ked_aupr>m1": ked["aupr"] > m1["aupr"],
        "ked_aupr>最强单体": ked["aupr"] > best_single_aupr,
        "ked_auc>最强单体": ked["auc"] > best_single_auc,
        "ked_auc>PC": ked["auc"] > pc["auc"],
        "ked_aupr>PC": ked["aupr"] > pc["aupr"],
    }


def roc_auc(scores, gold_set):
    """平均秩法 AUC（与 eval_comparison.py 一致）"""
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
    """块积分 AP（与 eval_comparison.py 一致）"""
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


def load_wide():
    rows = []
    with open(WIDE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    rows = load_wide()
    n = len(rows)
    keys = [(r["层组合"], r["源节点"], r["目标节点"]) for r in rows]
    gold_full = [int(r["金标准(1=专家认可)"]) for r in rows]
    gold_idx = [i for i, g in enumerate(gold_full) if g == 1]
    assert len(gold_idx) == 17, "金标准应为 17 条，实际 %d" % len(gold_idx)

    cols = {disp: rows[0][col] for disp, col in METHODS}  # sanity: 列存在性
    scores = {}
    for disp, col in METHODS:
        scores[disp] = [(keys[i], float(rows[i][col] or 0.0)) for i in range(n)]

    # ---------- 基线复现（全 17 条金标准） ----------
    base = {}
    gold_set_full = {keys[i] for i in gold_idx}
    for disp, _ in METHODS:
        base[disp] = {
            "auc": roc_auc(scores[disp], gold_set_full),
            "aupr": average_precision(scores[disp], gold_set_full),
        }

    # ---------- LOEO 逐折 ----------
    folds = []
    for lo in gold_idx:
        gold_set = {keys[i] for i in gold_idx if i != lo}
        m = {}
        for disp, _ in METHODS:
            m[disp] = {
                "auc": roc_auc(scores[disp], gold_set),
                "aupr": average_precision(scores[disp], gold_set),
            }
        folds.append({"edge": keys[lo], "idx": lo, "metrics": m,
                      "conc": conclusions(m)})

    # ---------- 输出 CSV ----------
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "left_out_combo", "left_out_source", "left_out_target"]
                   + ["%s_auc" % d for d, _ in METHODS]
                   + ["%s_aupr" % d for d, _ in METHODS]
                   + ["ked_aupr>m1", "ked_aupr>最强单体", "ked_auc>最强单体",
                      "ked_auc>PC", "ked_aupr>PC"])
        for k, fd in enumerate(folds, 1):
            w.writerow([k, fd["edge"][0], fd["edge"][1], fd["edge"][2]]
                       + ["%.4f" % fd["metrics"][d]["auc"] for d, _ in METHODS]
                       + ["%.4f" % fd["metrics"][d]["aupr"] for d, _ in METHODS]
                       + [int(v) for v in fd["conc"].values()])

    # ---------- 汇总 txt ----------
    lines = []
    lines.append("Leave-one-edge-out 敏感性分析（2026-10-02）")
    lines.append("候选=%d 金标准=17；每折留一改负，剩 16 正例重算 AUC/AUPR" % n)
    lines.append("")
    lines.append("== 基线复现（全 17 条，应与 significance_tests.txt 一致） ==")
    for disp, _ in METHODS:
        lines.append("  %-28s AUC=%.4f AUPR=%.4f"
                     % (disp, base[disp]["auc"], base[disp]["aupr"]))
    lines.append("")
    lines.append("== LOEO 分布（17 折） ==")
    for disp, _ in METHODS:
        aucs = [fd["metrics"][disp]["auc"] for fd in folds]
        auprs = [fd["metrics"][disp]["aupr"] for fd in folds]
        lines.append("  %-28s AUC min=%.4f mean=%.4f max=%.4f | AUPR min=%.4f mean=%.4f max=%.4f"
                     % (disp, min(aucs), sum(aucs) / len(aucs), max(aucs),
                        min(auprs), sum(auprs) / len(auprs), max(auprs)))
    lines.append("")
    lines.append("== 结论保持率（17 折中成立折数 / 17） ==")
    for name in folds[0]["conc"]:
        cnt = sum(1 for fd in folds if fd["conc"][name])
        lines.append("  %-20s %d/17 (%.0f%%)" % (name, cnt, 100.0 * cnt / 17))
    lines.append("")
    lines.append("== 单折明细（对 KED-均值 主配置 AUPR 影响最大的折） ==")
    ranked = sorted(folds, key=lambda fd: fd["metrics"][KED]["aupr"])
    for fd in ranked[:5]:
        e = fd["edge"]
        lines.append("  留出 %s -> %s (%s)：KED AUPR=%.4f（基线 %.4f），m1 AUPR=%.4f（基线 %.4f）"
                     % (e[1], e[2], e[0], fd["metrics"][KED]["aupr"], base[KED]["aupr"],
                        fd["metrics"][M1]["aupr"], base[M1]["aupr"]))
    lines.append("")
    lines.append("产物：loeo_sensitivity.csv（17 折 × %d 方法明细），口径与 eval_comparison.py 一致。"
                 % len(METHODS))
    txt = "\n".join(lines)
    open(OUT_TXT, "w", encoding="utf-8").write(txt)
    print(txt)
    print("\n已写:", OUT_CSV)
    print("已写:", OUT_TXT)


if __name__ == "__main__":
    main()
