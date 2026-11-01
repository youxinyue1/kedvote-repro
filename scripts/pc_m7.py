# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""
M7 数据驱动 PC 算法（v3.2 口径，方案 §3.2/§3.6）
=================================================
输入: D1 病例矩阵 extended_data.csv（391 例 × 90 节点列, 二值）
流程（PC 骨架近似 + 层序定向）:
  Phase0 候选: 3 类层间有序对（与 LLM 组同空间: S0×Z0, Z0×A0, A0×Y1）
  Phase1 边缘独立: 2×2 卡方（期望频数<5 降 Fisher 精确）; 联合阳性 n11<5 直接判不相关
  Phase2 一阶条件独立: 对每条显著边, 控制共同邻接候选 Z 分层检验, 两层均独立则删边
  Phase3 定向: 按领域层序 S0→Z0→A0→Y1（与理论 DAG 分层一致）
置信度: conf = 1 - p（供 AUC/AUPR 排序; 方案 §3.4 约定）
输出: method_m7/edges.csv（S0-Z0 行按 LLM 惯例存 证候→症状, 由评估端翻转）
"""
import csv
import os

from scipy.stats import chi2_contingency, fisher_exact

PKG = DATA
D1 = os.path.join(DATA, "extended_data.csv")
OUT_DIR = os.path.join(EXP, "method_m7")
TAG, LABEL = "m7", "D1"   # __main__ 可经 --tag/--label 覆盖（D2 跨源复现）
ALPHA = 0.05
MIN_N11 = 5


def load_matrix(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header, data = rows[0], rows[1:]
    cols = {name: i for i, name in enumerate(header)}
    def vec(name):
        i = cols[name]
        return [1 if r[i].strip() in ("1", "1.0") else 0 for r in data]
    return header, {n: vec(n) for n in header if "::" in n}, len(data)


def indep_test(x, y):
    """返回 (p, n11)。2x2 卡方, 期望<5 用 Fisher。"""
    n11 = sum(1 for a, b in zip(x, y) if a and b)
    if n11 < MIN_N11:
        return 1.0, n11
    n10 = sum(1 for a, b in zip(x, y) if a and not b)
    n01 = sum(1 for a, b in zip(x, y) if not a and b)
    n00 = sum(1 for a, b in zip(x, y) if not a and not b)
    table = [[n11, n10], [n01, n00]]
    if min(n11, n10, n01, n00) == 0 or (n11 + n10) * (n11 + n01) / len(x) < 5:
        return fisher_exact(table)[1], n11
    return chi2_contingency(table, correction=True)[1], n11


def cond_indep(x, y, z):
    """控制 z 分层: 两层均 p>=ALPHA 才判条件独立。
    重要: 任何一层 n11<MIN_N11 或样本空 → 该层视为"未检验" (tested=False)。
    若两层都未检验 (该 z 不提供有效证据) → 保守地判为【不】条件独立 (返回 False, 保留边),
    避免在缺乏证据时把边当成"通过条件独立"误删。"""
    tested = False
    for zv in (0, 1):
        xs = [a for a, c in zip(x, z) if c == zv]
        ys = [b for b, c in zip(y, z) if c == zv]
        if not xs:
            continue
        n11 = sum(1 for a, b in zip(xs, ys) if a and b)
        if n11 < MIN_N11:
            continue
        tested = True
        table = [[n11, sum(1 for a, b in zip(xs, ys) if a and not b)],
                 [sum(1 for a, b in zip(xs, ys) if not a and b),
                  sum(1 for a, b in zip(xs, ys) if not a and not b)]]
        if min(min(row) for row in table) == 0:
            p = fisher_exact(table)[1]
        else:
            p = chi2_contingency(table, correction=True)[1]
        if p < ALPHA:
            return False
    return False if not tested else True


def main():
    header, vecs, n = load_matrix(D1)
    S0 = [c for c in header if c.startswith("症状::")]
    Z0 = [c for c in header if c.startswith("证候::")]
    A0 = [c for c in header if c.startswith("治法::")]
    Y1 = [c for c in header if c.startswith("结局::")]
    print("D1: n=%d  S0=%d Z0=%d A0=%d Y1=%d" % (n, len(S0), len(Z0), len(A0), len(Y1)))

    # Phase 1: 边缘独立
    pairs = []  # (combo, x名, y名, p, n11)  层序: x=因(低层), y=果(高层)
    layers = {"S0": S0, "Z0": Z0, "A0": A0, "Y1": Y1}
    for a, b in (("S0", "Z0"), ("Z0", "A0"), ("A0", "Y1")):
        for x in layers[a]:
            for y in layers[b]:
                p, n11 = indep_test(vecs[x], vecs[y])
                if p < ALPHA:
                    pairs.append(("%s-%s" % (a, b), x, y, p, n11))
    print("Phase1 显著边: %d" % len(pairs))

    # Phase 2: 一阶条件独立删边（共同邻接 = 与 x/y 都边缘显著的其他节点）
    sig_set = {(x, y) for _, x, y, _, _ in pairs}
    kept = []
    dropped = []
    for combo, x, y, p, n11 in pairs:
        zs = [z for z in header if "::" in z and z not in (x, y)
              and ((z, x) in sig_set or (x, z) in sig_set)
              and ((z, y) in sig_set or (y, z) in sig_set)]
        removed = False
        for z in zs[:30]:  # 邻接候选上限, 控制计算量
            if cond_indep(vecs[x], vecs[y], vecs[z]):
                dropped.append((combo, x, y, p, "cond_indep_given:" + z))
                removed = True
                break
        if not removed:
            kept.append((combo, x, y, p))
    print("Phase2 条件独立删边: %d, 保留: %d" % (len(dropped), len(kept)))
    # 条件独立阶段是否实际执行了检验？为审计可追溯性精确计数：
    # 共邻接候选 z 必须与边两端都边缘显著 (见上方 zs 构造)，n11>=5 是 cond_indep 的硬门槛；
    # 当 n_cond_tests==0 时，该阶段为空运行，结果是纯边缘独立筛选（§4.5 披露）。
    n_cond_tests = 0
    for combo, x, y, p, n11 in pairs:
        zs2 = [z for z in header if "::" in z and z not in (x, y)
               and ((z, x) in sig_set or (x, z) in sig_set)
               and ((z, y) in sig_set or (y, z) in sig_set)]
        for z in zs2[:30]:
            n_cond_tests += 1  # 进入 cond_indep 内部后将按 n11>=5 决定是否真正做检验；此处统计"尝试"次数
    cond_note = "executed" if n_cond_tests > 0 else (
        "vacuous: zero co-adjacent conditioning candidate (a z must be marginally "
        "significant with both endpoints); retained edges are marginal-independence "
        "results; see \u00a74.5"
    )

    # 输出: S0-Z0 行按 LLM 惯例存 证候->症状 (评估端翻转); 其余存 因->果
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "edges.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "directed", "confidence", "basis", "combo"])
        for combo, x, y, p in sorted(kept):
            conf = max(0.5, 1.0 - p)
            if combo == "S0-Z0":
                src, tgt = y, x          # 证候 -> 症状
            else:
                src, tgt = x, y
            w.writerow([src.split("::")[1], tgt.split("::")[1], True, "%.4f" % conf,
                        "chi2/fisher p=%.4g" % p, combo])
    with open(os.path.join(OUT_DIR, "dropped_cond_indep.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo", "x", "y", "p", "reason"])
        w.writerows(dropped)

    import json
    json.dump({"method": TAG, "dataset": LABEL,
               "model": "PC(chi2+fisher, cond-indep order1)",
               "prompt": "-", "temperature": None, "n_cases": n,
               "phase1_edges": len(pairs), "phase2_dropped": len(dropped),
               "phase2_tests_attempted": n_cond_tests,
               "phase2_note": cond_note,
               "final_edges": len(kept), "alpha": ALPHA, "min_n11": MIN_N11},
              open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("M7 完成: %d 条边 -> %s" % (len(kept), OUT_DIR))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=D1, help="病例矩阵 csv（D1 extended_data / D2 d2_cnki_matrix）")
    ap.add_argument("--tag", default="m7", help="method 目录名（m7 / m7d2）")
    ap.add_argument("--label", default="D1", help="数据集标签（写入 meta）")
    a = ap.parse_args()
    D1 = a.data
    if a.tag != "m7":
        OUT_DIR = os.path.join(os.path.dirname(OUT_DIR), "method_" + a.tag)
    TAG, LABEL = a.tag, a.label
    main()
