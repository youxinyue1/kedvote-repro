# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""逐条数据（候选边）识别结果导出
================================================
输出两张表到 results/（供导师查看与后续作图）：
1. per_edge_predictions_wide.csv —— 宽表：每行一条候选边（282 条），
   列 = 层组合 | 源节点 | 目标节点 | 金标准(1/0) | 各方法置信度得分（未生成=0）
2. per_edge_predictions_long.csv —— 长表：每行 = 方法×候选边 的记录，
   列 = method | 候选边 | 金标准 | 该方法得分 | 是否生成(1/0)
口径与统一评估完全一致（eval_comparison 的 load_pred）。
"""
import os
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
import eval_comparison as ec


# 全部有效方法（m8 输出损坏 0 边，剔除；按类别排序展示）
METHODS_ORDER = [
    # 单体基线
    "m1", "m2", "m3", "m9", "m9b", "m9c", "m10", "m11", "m8b",
    "m4", "m4b",
    # B 轴消融
    "m5", "m5_role_off", "m5_cot_off", "m5_lex_off", "m5_fewshot_off", "m5_conf_off",
    # 投票
    "m6", "m6b",
    # KED-Vote
    "m12_c3_d3", "m12_c3_d1", "m12_c3_d2", "m12_c3_d4",
    "m12_c1_d3", "m12_c1_d1", "m12_c1_d2", "m12_c1_d4",
    "m12_c2_d3", "m12_c2_d1", "m12_c2_d2", "m12_c2_d4",
    "m12_c4_d3", "m12_c4_d1", "m12_c4_d4",
    # 数据驱动
    "m7", "m7d2", "m7d2_391",
]

NICE = {  # 方法显示名（同 comparison_summary）
    "m1": "Qwen2.5-14B", "m2": "Qwen2.5-7B", "m3": "Qwen2.5-3B",
    "m9": "R1-14B", "m9b": "R1-7B", "m9c": "R1-1.5B",
    "m10": "GLM4-9B", "m11": "Yi-34B", "m8b": "Baichuan2-7B",
    "m4": "扁仓14B", "m4b": "沈仲7B",
    "m5": "V2完整", "m5_role_off": "V2-角色", "m5_cot_off": "V2-CoT",
    "m5_lex_off": "V2-词汇", "m5_fewshot_off": "V2-示例", "m5_conf_off": "V2-置信度",
    "m6": "旧交集票", "m6b": "旧多数票",
    "m12_c3_d3": "KED-均值(主)", "m12_c3_d1": "KED-交集", "m12_c3_d2": "KED-多数票",
    "m12_c3_d4": "KED-rank", "m12_c1_d3": "KED-c1均值", "m12_c1_d1": "KED-c1交集",
    "m12_c1_d2": "KED-c1多数", "m12_c1_d4": "KED-c1rank",
    "m12_c2_d3": "KED-c2均值", "m12_c2_d1": "KED-c2交集", "m12_c2_d2": "KED-c2多数",
    "m12_c2_d4": "KED-c2rank", "m12_c4_d3": "KED-c4均值", "m12_c4_d1": "KED-c4交集",
    "m12_c4_d4": "KED-c4rank",
    "m7": "PC-D1", "m7d2": "PC-D2", "m7d2_391": "PC-D2旧",
}


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    layers = json.load(open(ec.LAYERS, encoding="utf-8"))
    cand = ec.build_candidates(layers)
    gold = {(c, (s, t)) for c, (s, t) in ec.load_gold()}

    preds = {m: ec.load_pred(m) for m in METHODS_ORDER}

    # ---- 宽表 ----
    wpath = os.path.join(RESULTS, "per_edge_predictions_wide.csv")
    with open(wpath, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["层组合", "源节点", "目标节点", "金标准(1=专家认可)"]
                   + [NICE[m] for m in METHODS_ORDER])
        for combo, (s, t) in cand:
            key = (combo, (s, t))
            row = [combo, s, t, 1 if key in gold else 0]
            row += ["%.4f" % preds[m].get(key, 0.0) for m in METHODS_ORDER]
            w.writerow(row)

    # ---- 长表 ----
    lpath = os.path.join(RESULTS, "per_edge_predictions_long.csv")
    with open(lpath, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "combo", "source", "target", "gold", "score", "generated"])
        for m in METHODS_ORDER:
            for combo, (s, t) in cand:
                key = (combo, (s, t))
                sc = preds[m].get(key, 0.0)
                w.writerow([m, combo, s, t,
                            1 if key in gold else 0,
                            "%.4f" % sc, 1 if sc > 0 else 0])

    # 摘要
    n_gold = sum(1 for c in cand if c in gold)
    print("宽表: %d 行(候选边) x %d 方法列 -> %s" % (len(cand), len(METHODS_ORDER), wpath))
    print("长表: %d 行(%d 方法 x %d 边) -> %s" % (len(METHODS_ORDER) * len(cand), len(METHODS_ORDER), len(cand), lpath))
    print("金标准正例: %d / %d" % (n_gold, len(cand)))
    # 抽查一条金标准边的各方法得分
    sample = [c for c in cand if c in gold][0]
    print("\n抽查金标准边 %s: " % str(sample))
    for m in ["m1", "m12_c3_d3", "m12_c3_d1", "m7"]:
        print("  %-14s %.4f" % (NICE[m], preds[m].get(sample, 0.0)))


if __name__ == "__main__":
    main()
