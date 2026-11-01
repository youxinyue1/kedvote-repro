# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""KED-Vote：知识增强异构专家投票（本文方法主体，v3.3 方案）
====================================================================
基座 Qwen2.5-14B (m1) + 异构专家团，四正交轴中的 C/D 两轴：

C 轴 · 专家团构成（4 组）：
  c1 同规模异构      [m1, m9, m4]        Qwen14 + DeepSeekR1-14 + 扁仓14（三个 14B 异构）
  c2 同家族异规模    [m1, m2, m3]        Qwen 家族 14B/7B/3B
  c3 通用+领域混合   [m1, m9, m10, m4]   主配置：通用3家族 + 领域1（4 成员）
  c4 专家数消减      [m1, m9]            2 员（配合 c1=3员、c3=4员 构成 2/3/4 规模线）

D 轴 · 投票策略（4 种）：
  d1 交集        全体成员都生成才保留
  d2 多数票      ≥ floor(n/2)+1 成员生成
  d3 置信度均值  并集，得分=全体成员置信度均值（缺失记 0）
  d4 rank 融合   并集，各成员内按置信度排名归一(1=最靠前)，缺失记 0，取均值

产出 method_m12_cX_dY/edges.csv（格式与单模型一致，直接进 eval_comparison）。
主方法 KED-Vote = c3 × 全部 D 轴（择优口径在评估报告中呈现）。
"""
import csv
import os
import json

EXP = EXP  # 见文件头路径配置

C_AXIS = {
    "c1": ["m1", "m9", "m4"],
    "c2": ["m1", "m2", "m3"],
    "c3": ["m1", "m9", "m10", "m4"],
    "c4": ["m1", "m9"],
}
D_AXIS = ["d1", "d2", "d3", "d4"]

DESC = {
    "c1": "同规模异构(14B×3: Qwen+DeepSeek+扁仓)",
    "c2": "同家族异规模(Qwen 14/7/3B)",
    "c3": "通用+领域混合(主配置: Qwen14+DeepSeek14+GLM9+扁仓14)",
    "c4": "专家数消减(2员: Qwen14+DeepSeek14)",
    "d1": "交集",
    "d2": "多数票",
    "d3": "置信度均值",
    "d4": "rank融合",
}


def load(m):
    """edges.csv -> {(combo, source, target): conf}，同键取最大置信度。"""
    d = {}
    p = os.path.join(EXP, "method_%s" % m, "edges.csv")
    if not os.path.exists(p):
        raise SystemExit("缺成员产物: %s" % p)
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        try:
            conf = float(r["confidence"])
        except (TypeError, ValueError):
            continue
        if conf <= 0:
            continue
        key = (r["combo"].strip(), r["source"].strip(), r["target"].strip())
        if key not in d or conf > d[key]:
            d[key] = conf
    return d


def rank_norm(d):
    """成员内按置信度排名归一化到 (0,1]：最高分=1.0。"""
    if not d:
        return {}
    order = sorted(d, key=lambda k: d[k])
    n = len(order)
    return {k: (i + 1) / n for i, k in enumerate(order)}


def vote(members, strategy):
    sets = {m: load(m) for m in members}
    ranks = {m: rank_norm(s) for m, s in sets.items()}
    keys = set()
    for s in sets.values():
        keys |= set(s)
    n = len(members)
    thr = n // 2 + 1
    out = {}
    for k in keys:
        votes = [sets[m][k] for m in members if k in sets[m]]
        nv = len(votes)
        if strategy == "d1":
            ok = nv == n
            score = sum(votes) / nv
        elif strategy == "d2":
            ok = nv >= thr
            score = sum(votes) / nv
        elif strategy == "d3":
            ok = nv >= 1
            score = sum(votes) / n          # 缺失记 0 后的均值
        else:  # d4
            ok = nv >= 1
            score = sum(ranks[m][k] for m in members if k in sets[m]) / n
        if ok and score > 0:
            out[k] = score
    return out, sets


def dump(name, rows, members, strategy):
    od = os.path.join(EXP, "method_%s" % name)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "edges.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "directed", "confidence", "basis", "combo"])
        for (combo, s, t), conf in sorted(rows.items()):
            w.writerow([s, t, True, "%.4f" % conf, "KED-Vote %s+%s" % (",".join(members), strategy), combo])
    meta = {"method": name, "model": "KED-Vote ensemble",
            "members": members, "strategy": strategy,
            "c_axis": DESC.get(name.split("_")[1] if "_" in name else "", ""),
            "d_axis": DESC.get(strategy, ""),
            "n_edges": len(rows)}
    json.dump(meta, open(os.path.join(od, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("%-14s %-12s 成员=%-28s 边=%d" % (name, strategy, ",".join(members), len(rows)))


def main():
    for cx, members in C_AXIS.items():
        for d in D_AXIS:
            rows, _ = vote(members, d)
            dump("m12_%s_%s" % (cx, d), rows, members, d)


if __name__ == "__main__":
    main()
