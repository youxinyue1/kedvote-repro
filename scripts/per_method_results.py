# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""每个算法单独导出逐条识别结果表（导师要求的"每算法一表"）
============================================================
输出 results/per_method/<方法名>_逐条结果.csv，每张表 282 行：
  序号 | 层组合 | 源节点 | 目标节点 | 金标准(1=专家认可,0=否) | 算法置信度得分 | 算法判定(生成/未生成)
口径与统一评估、per_edge_predictions 完全一致。
另生成索引 README，按类别列出全部表。
"""
import os
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
import eval_comparison as ec

OUT = os.path.join(RESULTS, "per_method")
os.makedirs(OUT, exist_ok=True)

METHODS_ORDER = [
    ("m1", "Qwen2.5-14B", "单体基线"),
    ("m2", "Qwen2.5-7B", "单体基线"),
    ("m3", "Qwen2.5-3B", "单体基线"),
    ("m9", "DeepSeek-R1-14B", "单体基线"),
    ("m9b", "DeepSeek-R1-7B", "单体基线"),
    ("m9c", "DeepSeek-R1-1.5B", "单体基线"),
    ("m10", "GLM-4-9B", "单体基线"),
    ("m11", "Yi-1.5-34B", "单体基线"),
    ("m8b", "Baichuan2-7B", "单体基线"),
    ("m4", "扁仓-14B", "中医领域模型"),
    ("m4b", "沈仲-7B", "中医领域模型"),
    ("m5", "Qwen2.5-14B V2完整提示", "B轴提示消融"),
    ("m5_role_off", "V2摘除角色锚定", "B轴提示消融"),
    ("m5_cot_off", "V2摘除CoT推理", "B轴提示消融"),
    ("m5_lex_off", "V2摘除词汇规范", "B轴提示消融"),
    ("m5_fewshot_off", "V2摘除few-shot", "B轴提示消融"),
    ("m5_conf_off", "V2摘除置信度锚定", "B轴提示消融"),
    ("m6", "旧版三模型交集投票", "投票对照"),
    ("m6b", "旧版三模型多数票", "投票对照"),
    ("m12_c3_d3", "KED-Vote 均值(主配置4员)", "本文方法KED-Vote"),
    ("m12_c3_d1", "KED-Vote 交集(主配置4员)", "本文方法KED-Vote"),
    ("m12_c3_d2", "KED-Vote 多数票(主配置4员)", "本文方法KED-Vote"),
    ("m12_c3_d4", "KED-Vote rank融合(主配置4员)", "本文方法KED-Vote"),
    ("m12_c1_d3", "KED-Vote c1三员均值", "本文方法KED-Vote"),
    ("m12_c1_d1", "KED-Vote c1三员交集", "本文方法KED-Vote"),
    ("m12_c1_d2", "KED-Vote c1三员多数票", "本文方法KED-Vote"),
    ("m12_c1_d4", "KED-Vote c1三员rank", "本文方法KED-Vote"),
    ("m12_c2_d3", "KED-Vote c2同家族均值", "本文方法KED-Vote"),
    ("m12_c2_d1", "KED-Vote c2同家族交集", "本文方法KED-Vote"),
    ("m12_c2_d2", "KED-Vote c2同家族多数票", "本文方法KED-Vote"),
    ("m12_c2_d4", "KED-Vote c2同家族rank", "本文方法KED-Vote"),
    ("m12_c4_d3", "KED-Vote c4两员均值", "本文方法KED-Vote"),
    ("m12_c4_d1", "KED-Vote c4两员交集", "本文方法KED-Vote"),
    ("m12_c4_d4", "KED-Vote c4两员rank", "本文方法KED-Vote"),
    ("m7", "PC算法 D1临床数据", "数据驱动对照"),
    ("m7d2", "PC算法 D2验案448例", "数据驱动对照"),
    ("m7d2_391", "PC算法 D2旧版391例", "数据驱动对照"),
]


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    layers = json.load(open(ec.LAYERS, encoding="utf-8"))
    cand = ec.build_candidates(layers)
    gold = {(c, (s, t)) for c, (s, t) in ec.load_gold()}

    idx_lines = ["# 每算法逐条识别结果（索引）\n",
                 "每张表 282 行 = 全部候选因果边。列：序号 | 层组合 | 源节点 | 目标节点 | "
                 "金标准(1=专家认可) | 置信度得分(0=未识别) | 判定。\n"]
    last_cat = None
    for mid, nice, cat in METHODS_ORDER:
        pred = ec.load_pred(mid)
        fname = "%s_逐条结果.csv" % nice.replace(" ", "")
        fpath = os.path.join(OUT, fname)
        with open(fpath, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["序号", "层组合", "源节点", "目标节点",
                        "金标准(1=专家认可)", "置信度得分", "算法判定"])
            for i, (combo, (s, t)) in enumerate(cand, 1):
                key = (combo, (s, t))
                sc = pred.get(key, 0.0)
                w.writerow([i, combo, s, t,
                            1 if key in gold else 0,
                            "%.4f" % sc,
                            "识别" if sc > 0 else "未识别"])
        if cat != last_cat:
            idx_lines.append("\n## %s\n" % cat)
            last_cat = cat
        # 统计该表摘要
        gen = sum(1 for c in cand if pred.get(c, 0.0) > 0)
        tp = sum(1 for c in cand if c in gold and pred.get(c, 0.0) > 0)
        idx_lines.append("- `%s`：生成 %d 条，命中金标准 %d/17 —— 文件 `%s`"
                         % (nice, gen, tp, fname))
        print("OK %-28s 生成%3d条 命中%2d/17" % (nice, gen, tp))

    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(idx_lines))
    print("\n共 %d 张表 -> %s" % (len(METHODS_ORDER), OUT))


if __name__ == "__main__":
    main()
