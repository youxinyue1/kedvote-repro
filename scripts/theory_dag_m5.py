# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""
M5 提示增强（V2=S_full）+ B 轴最小组件消融（role_off / cot_off）
================================================================
v3.2 口径：模型 qwen2.5:14b (Q4_K_M)，temperature=0.0，节点集/评估与 M1 完全一致。
- m5        : V2 完整结构 = V0 + 角色锚定强化 + 词汇规范内联 + 同义合并 + CoT + few-shot
- m5_role_off: 摘除角色锚定（"资深主任医师+指南"→ 通用助手），其余 V2 组件保留
- m5_cot_off : 摘除 CoT 三步推理指令，其余 V2 组件保留
few-shot 示例全部为外感病非肺癌节点对，与金标准 17 条零重合（无泄露声明 §6）。
用法: python3 theory_dag_m5.py --method m5 [--task S0-Z0] ...
"""
import os
import re
import csv
import json
import time
import argparse
import urllib.request
import urllib.error

BASE = os.path.dirname(os.path.abspath(__file__))
LAYERS_DEF = os.path.join(DATA, "node_layers.json")
OLLAMA = "http://127.0.0.1:11434"
MODEL_ID = "qwen2.5:14b"

METHODS = {
    "m5":              "V2 S_full",
    "m5_role_off":     "V2 minus role",
    "m5_cot_off":      "V2 minus CoT",
    "m5_lex_off":      "V2 minus lexicon(词汇规范/同义合并)",
    "m5_fewshot_off":  "V2 minus few-shot 示例",
    "m5_conf_off":     "V2 minus 置信度锚定(不要求0-1置信度列)",
}
MAX_RETRY = 3
TIMEOUT = 600
MAX_TOKENS = 768
TASKS = ["S0-Z0", "Z0-A0", "A0-Y1"]

PREFIXES = ("证候::", "症状::", "治法::", "结局::", "舌诊::", "脉诊::")
SUFFIXES = ("证候", "症状", "治法")
SYNONYM = {
    "症状改善评分": "症状改善", "症状减轻": "症状改善",
    "证候改善总分": "二诊改善", "证候改善评分": "二诊改善",
    "症状改善": "症状改善", "二诊改善": "二诊改善",
    "食欲不振": "食欲", "食少纳呆": "食欲", "纳呆食少": "食欲",
}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def norm_name(x):
    x = x.strip()
    for p in PREFIXES:
        if x.startswith(p):
            x = x[len(p):]
            break
    for s in SUFFIXES:
        if x.endswith(s) and len(x) > len(s):
            x = x[:-len(s)]
            break
    return SYNONYM.get(x, x)


def to_conf(s):
    """从文本段提取 0-1 置信度。容忍 '85%' / '85％' → 0.85；'0.85' → 0.85；非数字返 None。"""
    if s is None:
        return None
    raw = str(s).strip()
    has_pct = ("%" in raw) or ("％" in raw)
    m = re.findall(r"\d+\.?\d*", raw.replace("％", "%").replace("%", ""))
    if not m:
        return None
    try:
        v = float(m[0])
    except ValueError:
        return None
    if has_pct and v > 1.0:  # '85%' → 85 → 0.85
        v = v / 100.0
    return max(0.0, min(1.0, v))


def call_llm(system, user):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0, "max_tokens": MAX_TOKENS, "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    last_err = None
    for _ in range(MAX_RETRY):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = "%s %s" % (e.code, e.read(300).decode("utf-8", "ignore"))
            time.sleep(8)
        except Exception as e:
            last_err = repr(e)
            time.sleep(8)
    raise RuntimeError("LLM调用失败: %s" % last_err)


def parse_lines(text, src_ok, tgt_ok):
    edges = []
    src_ok = {norm_name(x) for x in src_ok}
    tgt_ok = {norm_name(x) for x in tgt_ok}
    for raw in text.splitlines():
        ln = raw.strip()
        if ln.startswith("```"):
            continue
        ln = ln.lstrip("-*0123456789.、)） \t")
        if not ln or len(ln) < 6:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 2:
            parts = [p.strip() for p in ln.replace("→", "|").replace("->", "|").split("|")]
        if len(parts) < 2:
            continue
        src, tgt = norm_name(parts[0]), norm_name(parts[1])
        basis = parts[2][:60] if len(parts) >= 3 else ""
        conf = to_conf(parts[3]) if len(parts) >= 4 else None
        if src in src_ok and tgt in tgt_ok:
            edges.append((src, tgt, basis, conf))
        elif tgt in src_ok and src in tgt_ok:
            edges.append((tgt, src, basis, conf))
    return edges


# ------------------------------------------------------------- V2 组件模板
ROLE_ON = """你是从事肺癌中医诊治 30 年的资深主任医师, 兼任中华中医药学会肿瘤分会常委, 精通《肺癌中医诊疗指南》与中医辨证理论体系。"""
ROLE_OFF = """你是一个乐于助人的通用 AI 助手。"""

COT_ON = """【推理要求】先在内部完成三步推理再输出:
(1) 该证候的核心病机是什么;
(2) 候选症状中哪些由该病机直接派生、且为该证候与其余证候的鉴别要点;
(3) 剔除气虚/血虚/阴虚等多证候共有的泛化症状后, 保留 1-2 个特征性主症。"""

LEXICON = """【词汇规范】
- 输出的症状名/治法名必须一字不差取自候选清单; 清单外词汇一律不得输出。
- 同义合并: 纳呆/食少纳呆/食欲不振 统一记"食欲"; 症状减轻/症状改善评分 统一记"症状改善"。"""

FEWSHOT = """【输出示例】(示例为外感病非肺癌内容, 仅示范格式与置信度用法)
风寒束表 | 恶寒 | 恶寒为风寒束表特征性主症 | 0.95
风寒束表 | 咳嗽 | 风寒束肺肺气上逆, 非特征性但常见 | 0.70"""


def build_sys(task, variant):
    role = ROLE_OFF if variant == "m5_role_off" else ROLE_ON
    cot = "" if variant == "m5_cot_off" else COT_ON
    lex = "" if variant == "m5_lex_off" else LEXICON
    few = "" if variant in ("m5_cot_off", "m5_fewshot_off") else FEWSHOT
    if task == "S0-Z0":
        body = """完成"症状->证候"的辨证知识归纳。
【硬性规则】
- 只列特征性核心主症(1-2个): 该证候成立时几乎必见、最能与其它证候鉴别。
- 严禁列多证候共有的泛化症状(如乏力/发热), 除非它正是该证诊断关键。
- 必须逐行输出, 禁止省略、禁止写"等"; 未列出的候选症状即视为不存在。
- 【置信度】每行末尾给 0-1 数字, 表示该边在真实临床成立的把握。
【输出格式】每行严格为: 证候名 | 症状名 | 辨证依据(≤25字) | 置信度(0-1)
只输出这些行, 不要任何解释、编号或多余文字。"""
    elif task == "Z0-A0":
        body = """完成"证候->治法"的证法相符知识归纳。
【硬性规则】
- 只给一个标准对证治法(该证候确立后临床首选的规范治法)。
- 治法名必须一字不差取自候选治法; 若候选无对应治法, 输出"无"。
- 必须逐行输出, 禁止省略。
- 【置信度】每行末尾给 0-1 数字, 表示该证-法配伍成立的把握。
【输出格式】每行严格为: 证候名 | 治法名 | 依据(≤20字) | 置信度(0-1)
只输出这些行, 不要任何解释、编号或多余文字。"""
    else:
        body = """完成"治法->结局"的疗效知识归纳。
【硬性规则】
- 列出该治法在肺癌治疗中公认能直接改善的结局(从候选中选); 每个结局一行。
- 若无直接作用, 输出"无"; 宁缺毋滥。
- 必须逐行输出, 禁止省略、禁止写"等"。
- 【置信度】每行末尾给 0-1 数字, 表示该治法能直接改善该结局的把握。
【输出格式】每行严格为: 治法名 | 结局名 | 依据(≤20字) | 置信度(0-1)
只输出这些行, 不要任何解释、编号或多余文字。"""
    if variant == "m5_conf_off":
        # 摘除置信度锚定：删【置信度】规则行 + 输出格式去置信度列
        body = "\n".join(l for l in body.split("\n") if "【置信度】" not in l)
        body = body.replace(" | 置信度(0-1)", "")
    blocks = [role, "任务: " + body]
    if cot:
        blocks.append(cot)
    blocks.append(lex)
    if few:
        blocks.append(few)
    return "\n\n".join(b for b in blocks if b)


def build_user(task, key, layers, variant="m5"):
    conf_col = "" if variant == "m5_conf_off" else " | 置信度"
    if task == "S0-Z0":
        cands = "、".join(norm_name(x) for x in layers["S0-症状"])
        return "【证候】%s\n【候选症状】%s\n请输出 证候名 | 症状名 | 辨证依据%s:" % (key, cands, conf_col)
    if task == "Z0-A0":
        cands = "、".join(norm_name(x) for x in layers["A0-治法"])
        return "【证候】%s\n【候选治法】%s\n请输出 证候名 | 治法名 | 依据%s:" % (key, cands, conf_col)
    cands = "、".join(norm_name(x) for x in layers["Y1-结局"])
    return "【治法】%s\n【候选结局】%s\n请输出 治法名 | 结局名 | 依据%s:" % (key, cands, conf_col)


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS.keys()))
    ap.add_argument("--root", default=EXP, help="方法目录的父目录; 默认 $KED_ROOT/exp")
    ap.add_argument("--layers", default=LAYERS_DEF)
    ap.add_argument("--task", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layers = json.load(open(args.layers, encoding="utf-8"))
    method_dir = os.path.join(args.root, "method_%s" % args.method)
    raw_dir = os.path.join(method_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_csv = os.path.join(method_dir, "edges.csv")

    tasks = [args.task] if args.task else TASKS
    defs = {
        "S0-Z0": (layers["Z0-证候"], layers["S0-症状"]),
        "Z0-A0": (layers["Z0-证候"], layers["A0-治法"]),
        "A0-Y1": (layers["A0-治法"], layers["Y1-结局"]),
    }

    done = {}
    if os.path.exists(out_csv) and not args.force:
        for r in csv.DictReader(open(out_csv, encoding="utf-8-sig")):
            done.setdefault(r["combo"], set()).add(r["source"])

    new_rows = []
    t_start = time.time()
    no_conf = 0
    for task in tasks:
        src_nodes, tgt_nodes = defs[task]
        log("=" * 60)
        log("任务 %s: %d 个源节点" % (task, len(src_nodes)))
        for key in src_nodes:
            nkey = norm_name(key)
            if not args.force and task in done and nkey in done[task]:
                log("  [%s] %s 已缓存, 跳过" % (task, nkey))
                continue
            sys_p = build_sys(task, args.method)
            user = build_user(task, nkey, layers, variant=args.method)
            log("  [%s] %s ..." % (task, nkey))
            t0 = time.time()
            text = call_llm(sys_p, user)
            with open(os.path.join(raw_dir, "%s_%s.txt" % (task, nkey)), "w", encoding="utf-8") as f:
                f.write(text)
            edges = parse_lines(text, {key}, set(tgt_nodes))
            if not edges:
                time.sleep(2)
                text = call_llm(sys_p, user)
                with open(os.path.join(raw_dir, "%s_%s.txt" % (task, nkey)), "w", encoding="utf-8") as f:
                    f.write(text)
                edges = parse_lines(text, {key}, set(tgt_nodes))
            log("    返回%d条边, wall=%.1fs" % (len(edges), time.time() - t0))
            for s, t, b, c in edges:
                if c is None:
                    no_conf += 1
                new_rows.append({"source": s, "target": t, "directed": True,
                                 "confidence": c if c is not None else 1.0,
                                 "basis": b, "combo": task})
            time.sleep(1)

    meta = {"method": args.method, "model": "qwen2.5:14b (Q4_K_M)",
            "ollama_id": MODEL_ID, "prompt": METHODS[args.method],
            "temperature": 0.0, "wall_s": round(time.time() - t_start, 1),
            "new_edges": len(new_rows), "missing_conf_defaulted_1": no_conf}
    json.dump(meta, open(os.path.join(method_dir, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    if new_rows:
        fresh = not os.path.exists(out_csv)
        with open(out_csv, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "target", "directed",
                                              "confidence", "basis", "combo"])
            if fresh:
                w.writeheader()
            for r in new_rows:
                w.writerow(r)
        log("已追加 %d 条 -> %s" % (len(new_rows), out_csv))
    log("method_%s 完成: %d 条边, wall=%.0fs"
        % (args.method, len(new_rows), meta["wall_s"]))


if __name__ == "__main__":
    main()
