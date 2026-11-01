# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""M1-M4 V0 逐节点回忆 · 统一评估（本地运行）
============================================
对照方案 §3.4 输出 comparison_summary.csv（candidate-set 口径先行）：
  列: method, model, space, generated_edges, tp, fp, fn, precision, recall_17, recall_15, f1, auc, aupr
  - 候选空间(candidate-set)：标准节点集 S0/Z0/A0/Y1 内 5 类层间有序对
    (S0→Z0, S0→Y1, Z0→A0, Z0→Y1, A0→Y1)；未生成边 conf=0。
  - LLM 询问仅 3 类(理论层负责)；S0→Y1 / Z0→Y1 两对金标准由数据层承担，
    同时给出 recall_17(全金标准) 与 recall_15(理论层负责口径) 两种召回。
用法: python .server_bridge/eval_comparison.py [--space candidate-set]
产物(写到结果根目录 eval/):
  eval/comparison_summary.csv, eval/comparison_report.md
"""
import os
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = DATA
GT = os.path.join(DATA, "ground_truth_aligned.csv")
LAYERS = os.path.join(DATA, "node_layers.json")

EXP = EXP  # 见文件头路径配置
EVAL = os.path.join(EXP, "eval")
os.makedirs(EVAL, exist_ok=True)

METHODS = {"m1": "qwen2.5:14b (Q4_K_M)", "m2": "qwen2.5:7b (Q4_K_M)",
           "m3": "qwen2.5:3b (Q4_K_M)",
           "m4": "biancang-14b (Q8_0)", "m4b": "shizhengpt-7b (Q8_0)",
           "m8": "baichuan2-13b-chat (Q4_0)", "m8b": "baichuan2-7b-chat (Q4_0)",
           "m9": "deepseek-r1-14b (Q4_K_M)", "m9b": "deepseek-r1-7b (Q4_K_M)",
           "m9c": "deepseek-r1-1.5b (Q4_K_M)", "m10": "glm4-9b-chat (Q4_K_M)",
           "m11": "yi-1.5-34b-chat (Q4_0)",
           "m12_c1_d1": "KED-Vote c1(同规模异构14B×3) d1交集",
           "m12_c1_d2": "KED-Vote c1(同规模异构14B×3) d2多数票",
           "m12_c1_d3": "KED-Vote c1(同规模异构14B×3) d3置信度均值",
           "m12_c1_d4": "KED-Vote c1(同规模异构14B×3) d4rank融合",
           "m12_c2_d1": "KED-Vote c2(同家族异规模Qwen14/7/3) d1交集",
           "m12_c2_d2": "KED-Vote c2(同家族异规模Qwen14/7/3) d2多数票",
           "m12_c2_d3": "KED-Vote c2(同家族异规模Qwen14/7/3) d3置信度均值",
           "m12_c2_d4": "KED-Vote c2(同家族异规模Qwen14/7/3) d4rank融合",
           "m12_c3_d1": "KED-Vote c3(通用+领域混合主配置4员) d1交集",
           "m12_c3_d2": "KED-Vote c3(通用+领域混合主配置4员) d2多数票",
           "m12_c3_d3": "KED-Vote c3(通用+领域混合主配置4员) d3置信度均值",
           "m12_c3_d4": "KED-Vote c3(通用+领域混合主配置4员) d4rank融合",
           "m12_c4_d1": "KED-Vote c4(专家数2员) d1交集",
           "m12_c4_d2": "KED-Vote c4(专家数2员) d2多数票",
           "m12_c4_d3": "KED-Vote c4(专家数2员) d3置信度均值",
           "m12_c4_d4": "KED-Vote c4(专家数2员) d4rank融合",
           "m5": "qwen2.5:14b V2 S_full", "m5_role_off": "V2 minus role",
           "m5_cot_off": "V2 minus CoT",
           "m5_lex_off": "V2 minus 词汇规范", "m5_fewshot_off": "V2 minus few-shot",
           "m5_conf_off": "V2 minus 置信度锚定",
           "m6": "M1+M2+M4 交集投票", "m6b": "M1+M2+M4 多数票投票",
           "m7": "PC (chi2+fisher, D1)", "m7d2": "PC (chi2+fisher, D2 448例扩样)",
           "m7d2_391": "PC (chi2+fisher, D2 391例旧版)"}

# ------------------------------------------------------------ 候选空间
PREFIX = {"S0": "症状::", "Z0": "证候::", "A0": "治法::", "Y1": "结局::"}


def build_candidates(layers):
    """候选空间。node_layers 节点名本身已带层前缀（如 '症状::乏力'），直接引用。"""
    S0, Z0, A0, Y1 = layers["S0-症状"], layers["Z0-证候"], layers["A0-治法"], layers["Y1-结局"]
    combos = [("S0-Z0", S0, Z0), ("S0-Y1", S0, Y1), ("Z0-A0", Z0, A0),
              ("Z0-Y1", Z0, Y1), ("A0-Y1", A0, Y1)]
    cand = []
    for combo, ls, lt in combos:
        for s in ls:
            for t in lt:
                cand.append((combo, (s, t)))
    return cand


def load_gold():
    """返回 {(combo,(source,target))}, (source,target) 已带层前缀、方向与 GT 一致"""
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


def load_pred(method):
    """edges.csv -> {(combo,(source,target)): conf}；conf<=0 的行视为未生成剔除。
    S0-Z0 行存的是 证候->症状，需翻转为 GT 的 症状->证候。"""
    csvp = os.path.join(EXP, "method_%s" % method, "edges.csv")
    d = {}
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


# ------------------------------------------------------------ 指标
def prf(pred, gold, layers, gt_resp_combo):
    """P/R/F1：pred 集合(含前缀对) 内金标准比对。返回 per-combo 与合计。
    per_layer 行覆盖全部 5 类层组合（含数据层负责的 S0-Y1/Z0-Y1，LLM/PC 均无预测时 P 记 0）"""
    rows = []
    all_tp = all_fp = all_fn = 0
    combos = ["S0-Z0", "Z0-A0", "A0-Y1", "S0-Y1", "Z0-Y1"]
    for combo in combos:
        g = {(s, t) for c, (s, t) in gold if c == combo}
        p = {(s, t) for c, (s, t) in pred.keys() if c == combo}
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        pc = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / len(g) if g else float("nan")
        fc = 2 * pc * rc / (pc + rc) if (pc + rc) else 0.0
        rows.append((combo, len(p), len(g), tp, fp, fn, pc, rc, fc))
    # 口径2 recall：仅理论层负责的 15 条
    g15 = len(gt_resp_combo)
    prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    rec17 = all_tp / len(gold) if gold else 0.0
    rec15 = all_tp / g15 if g15 else 0.0
    f1_17 = 2 * prec * rec17 / (prec + rec17) if (prec + rec17) else 0.0
    f1_15 = 2 * prec * rec15 / (prec + rec15) if (prec + rec15) else 0.0
    return dict(per_layer=rows, n_pred=len(pred), tp=all_tp, fp=all_fp, fn=all_fn,
                prec=prec, rec17=rec17, rec15=rec15, f1_17=f1_17, f1_15=f1_15)


def roc_auc(scores, gold_set):
    """scores: [(pair_key, conf)] 覆盖全部候选; gold_set 为正例键集。
    ties 取平均秩法: AUC = (mean_rank_pos - (P+1)/2) / Nneg"""
    order = sorted(scores, key=lambda x: x[1])
    n = len(order)
    # 计算每元素平均秩（同分取平均）
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and order[j + 1][1] == order[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based 平均秩
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
    """AP=Σ P@k·ΔR，按块积分处理同分（近似 sklearn average_precision）"""
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


def eval_method(method, cand, gold, gold_set, gt_resp_combo):
    pred = load_pred(method)
    # 生成边（>=1 的行，去重后）必须都在候选空间内（模型仅从候选输出）
    # 落盘 rows 可能在 S0-Z0 swap 后依旧匹配候选键
    p = prf(pred, gold, cand, gt_resp_combo)
    scores = [(key, conf) for (key, conf) in pred.items()]  # 生成边
    # 补全候选空间 conf=0
    gold_keys = {(c, (s, t)) for c, (s, t) in gold}
    cand_keys = set(cand)
    full = [(k, pred.get(k, 0.0)) for k in cand_keys]
    auc = roc_auc(full, {k for k in gold_keys if k in cand_keys})
    aupr = average_precision(full, {k for k in gold_keys if k in cand_keys})
    return dict(method=method, model=METHODS[method], generated=len(pred),
                per_layer=p["per_layer"], tp=p["tp"], fp=p["fp"], fn=p["fn"],
                prec=p["prec"], rec17=p["rec17"], rec15=p["rec15"],
                f1_17=p["f1_17"], f1_15=p["f1_15"], auc=auc, aupr=aupr)


def main():
    layers = json.load(open(LAYERS, encoding="utf-8"))
    cand = build_candidates(layers)
    gold = load_gold()
    gold_keys = {(c, (s, t)) for c, (s, t) in gold}
    gt_resp = {(c, (s, t)) for c, (s, t) in gold if c in ("S0-Z0", "Z0-A0", "A0-Y1")}

    lines = ["# M1-M4b V0 逐节点回忆 · 统一评估（candidate-set, v3.2 Ollama 口径）\n"]
    lines.append("金标准共 %d 条；理论层负责 %d 条（S0-Z0/Z0-A0/A0-Y1），"
                 "S0-Y1/Z0-Y1 2 条由数据层单独承担。候选空间共 %d 个有序对。\n"
                 % (len(gold), len(gt_resp), len(cand)))

    print("DEBUG gold=%d cand=%d gold_in_cand=%d"
          % (len(gold), len(cand), len(gold_keys & set(cand))))
    summary = []
    for m in ["m1", "m2", "m3", "m4", "m4b", "m8", "m8b", "m9", "m9b", "m9c", "m10", "m11",
              "m5", "m5_role_off", "m5_cot_off", "m5_lex_off", "m5_fewshot_off", "m5_conf_off", "m6", "m6b",
              "m12_c1_d1", "m12_c1_d2", "m12_c1_d3", "m12_c1_d4",
              "m12_c2_d1", "m12_c2_d2", "m12_c2_d3", "m12_c2_d4",
              "m12_c3_d1", "m12_c3_d2", "m12_c3_d3", "m12_c3_d4",
              "m12_c4_d1", "m12_c4_d2", "m12_c4_d3", "m12_c4_d4",
              "m7", "m7d2", "m7d2_391"]:
        r = eval_method(m, cand, gold, gold_keys, gt_resp)
        summary.append(r)
        lines.append("### %s (%s)\n" % (r["method"], r["model"]))
        lines.append("生成边=%d (去重后), TP=%d FP=%d FN=%d\n" % (r["generated"], r["tp"], r["fp"], r["fn"]))
        lines.append("| 层组合 | 预测 | GT | TP | FP | FN | P | R | F1 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for combo, np_, ng, tp, fp, fn, pc, rc, fc in r["per_layer"]:
            lines.append("| %s | %d | %d | %d | %d | %d | %s | %s | %s |"
                         % (combo, np_, ng, tp, fp, fn,
                            "%.3f" % pc, "—" if rc != rc else "%.3f" % rc,
                            "—" if rc != rc else "%.3f" % fc))
        lines.append("\nPrecision=%.3f  Recall17=%.3f  Recall15=%.3f  F1_17=%.3f  F1_15=%.3f\n"
                     % (r["prec"], r["rec17"], r["rec15"], r["f1_17"], r["f1_15"]))
        lines.append("AUC=%.4f  AUPR=%.4f  (候选空间 %d 对, 正例 %d)\n"
                     % (r["auc"], r["aupr"], len(cand), len(gold_keys & set(cand))))

    # comparison_summary.csv
    with open(os.path.join(EVAL, "comparison_summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "model", "prompt_version", "space", "generated_edges",
                    "tp", "fp", "fn", "precision", "recall_17", "recall_15", "f1_17", "f1_15",
                    "auc", "aupr"])
        for r in summary:
            w.writerow([r["method"], r["model"], "V0+conf", "candidate-set", r["generated"],
                        r["tp"], r["fp"], r["fn"], "%.4f" % r["prec"], "%.4f" % r["rec17"],
                        "%.4f" % r["rec15"], "%.4f" % r["f1_17"], "%.4f" % r["f1_15"],
                        "%.4f" % r["auc"], "%.4f" % r["aupr"]])

    # comparison_per_combo.csv —— 分类别（层组合）P/R/F1 明细（v3.3 新增，导师要求）
    with open(os.path.join(EVAL, "comparison_per_combo.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "model", "combo", "n_pred", "n_gt", "tp", "fp", "fn",
                    "precision", "recall", "f1"])
        for r in summary:
            for combo, np_, ng, tp, fp, fn, pc, rc, fc in r["per_layer"]:
                w.writerow([r["method"], r["model"], combo, np_, ng, tp, fp, fn,
                            "%.4f" % pc, "" if rc != rc else "%.4f" % rc,
                            "" if rc != rc else "%.4f" % fc])
    open(os.path.join(EVAL, "comparison_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("summary 已写:", os.path.join(EVAL, "comparison_summary.csv"))
    for r in summary:
        print("%s gen=%-4d P=%.3f R15=%.3f R17=%.3f F1_15=%.3f AUC=%.4f AUPR=%.4f"
              % (r["method"], r["generated"], r["prec"], r["rec15"], r["rec17"],
                 r["f1_15"], r["auc"], r["aupr"]))


if __name__ == "__main__":
    main()
