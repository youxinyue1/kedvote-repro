# -*- coding: utf-8 -*-
# ---- 路径配置（复现仓库统一入口，可用环境变量覆盖）----
import os as _os
ROOT = _os.environ.get("KED_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.environ.get("KED_DATA", _os.path.join(ROOT, "data"))
EXP = _os.environ.get("KED_EXP", _os.path.join(ROOT, "exp"))
RESULTS = _os.path.join(ROOT, "results")
# -------------------------------------------------------

"""
理论DAG生成 · V0 逐节点询问执行器（v3.2 实验脚本 M1-M4b）
==========================================================
在 v3.1 基础上适配 Ollama 统一推理入口（2026-10-01，V100 服务器）：
  1. 全部模型走本机 Ollama http://127.0.0.1:11434/v1/chat/completions
     （OpenAI 兼容端点，仅标准库，无第三方依赖）；
  2. 模型矩阵更新（v3.2）：
     m1=qwen2.5:14b(Q4_K_M)  m2=qwen2.5:7b  m3=qwen2.5:3b
     m4=biancang-14b(Q8_0 扁仓,与qwen2.5:14b同底座)  m4b=shizhengpt-7b(Q8_0 沈仲)
  3. V0 提示词每行追加「0-1 置信度」输出字段（方案 §3.4/§8 口径）。
  4. 输出按 method_mN/edges.csv 分目录落盘, 列: source,target,directed,confidence,basis,combo；
     原始回复存 method_mN/raw/，运行信息写 meta.json，可断点续跑/审计。

用法:
  python3 theory_dag_exp.py --method m1 [--root $KED_EXP] [--layers $KED_DATA/node_layers.json]
                           [--task S0-Z0|Z0-A0|A0-Y1] [--force]
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
# 方法 -> (base_url, ollama模型注册名, 显示名)
MODELS = {
    "m1":  (OLLAMA, "qwen2.5:14b",   "qwen2.5-14b-instruct (Q4_K_M)"),
    "m2":  (OLLAMA, "qwen2.5:7b",    "qwen2.5-7b-instruct (Q4_K_M)"),
    "m3":  (OLLAMA, "qwen2.5:3b",    "qwen2.5-3b-instruct (Q4_K_M)"),
    "m4":  (OLLAMA, "biancang-14b",  "BianCang-Qwen2.5-14B-Instruct (Q8_0)"),
    "m4b": (OLLAMA, "shizhengpt-7b", "ShizhenGPT-7B (Q8_0)"),
    # v3.3 扩容：横向对比（导师要求 ≥2 个非千问家族）
    "m8":  (OLLAMA, "baichuan2:13b",  "Baichuan2-13B-Chat (Q4_0)"),
    "m8b": (OLLAMA, "baichuan2:7b",   "Baichuan2-7B-Chat (Q4_0)"),
    "m9":  (OLLAMA, "deepseek-r1:14b",   "DeepSeek-R1-Distill-Qwen-14B (Q4_K_M)"),
    "m9b": (OLLAMA, "deepseek-r1:7b",    "DeepSeek-R1-Distill-Qwen-7B (Q4_K_M)"),
    "m9c": (OLLAMA, "deepseek-r1:1.5b",  "DeepSeek-R1-Distill-Qwen-1.5B (Q4_K_M)"),
    "m10": (OLLAMA, "glm4:9b",           "GLM-4-9B-Chat (Q4_K_M)"),
    # v3.3b：Baichuan2-13B 两个第三方 GGUF 输出损坏且 convert 转换失败，替换为 Yi-1.5-34B（Ollama 官方库直拉，01.AI 家族）
    "m11": (OLLAMA, "yi:34b",            "Yi-1.5-34B-Chat (Q4_0)"),
}
# per-model 覆盖：R1 蒸馏模型带思维链，需更大 token 预算与超时
MODEL_CFG = {
    "deepseek-r1:14b":  {"max_tokens": 3072, "timeout": 2400},
    "deepseek-r1:7b":   {"max_tokens": 3072, "timeout": 2400},
    "deepseek-r1:1.5b": {"max_tokens": 3072, "timeout": 1800},
    "glm4:9b":          {"max_tokens": 1024, "timeout": 1200},
    "baichuan2:13b":    {"max_tokens": 1024, "timeout": 1200},
    "yi:34b":           {"max_tokens": 1024, "timeout": 1200},
}
MAX_RETRY = 3
TIMEOUT = 900          # Q8 大模型首次加载较慢，放宽
MAX_TOKENS = 512
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


def call_llm(base_url, model_id, system, user, temperature=0.0):
    cfg = MODEL_CFG.get(model_id, {})
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": cfg.get("max_tokens", MAX_TOKENS),
        "stream": False,
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            with urllib.request.urlopen(req, timeout=cfg.get("timeout", TIMEOUT)) as r:
                d = json.loads(r.read().decode("utf-8"))
            content = d["choices"][0]["message"]["content"]
            # R1 系思维链：剥 <think>...</think>，仅保留正文（无标签则原样）
            if "</think>" in content:
                content = content.split("</think>", 1)[1]
            return content.strip()
        except urllib.error.HTTPError as e:
            last_err = "%s %s" % (e.code, e.read(300).decode("utf-8", "ignore"))
            time.sleep(8)
        except Exception as e:
            last_err = repr(e)
            time.sleep(8)
    raise RuntimeError("LLM调用失败(%s): %s" % (model_id, last_err))


def parse_lines(text, src_ok, tgt_ok):
    """解析 V0+置信度 输出。每行: 源 | 目标 | 依据 | 置信度(0-1)。
    依据/置信度缺省容忍：缺依据记空、缺置信度记 1.0（缺失时按置信 1 判定成立）。
    返回 [(src, tgt, basis, conf)]；conf 为 float 或 None（记成立但无分，需外层定 1.0）。
    """
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


def fallback_parse(text, key, tgt_ok):
    """格式 C 兼容（Baichuan2 类自由格式）：『目标名 | 分数』逐行列表。
    仅当主解析 0 条时启用；分数 >=0.5 视为判定成立（与提示词"未列出的候选即不存在"
    的精神对齐——模型对不成立的候选给出 0.05~0.20 的低分，一律不收）。"""
    edges = []
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != 2:
            continue
        t = norm_name(parts[0])
        if t not in {norm_name(x) for x in tgt_ok}:
            continue
        try:
            c = float(parts[1])
        except ValueError:
            continue
        if 0.0 <= c <= 1.0 and c >= 0.5:
            edges.append((norm_name(key), t, "格式兼容解析(阈值0.5)", c))
    return edges


# ------------------------------------------------------------------ V0 prompts
def ask_s2z(layers, key):
    syms = "、".join(norm_name(x) for x in layers["S0-症状"])
    sys_p = """你是中医肺癌辨证论治领域专家, 依据《肺癌中医诊疗指南》及中医辨证理论, 完成"症状->证候"的辨证知识归纳。

【硬性规则】
- 只列出【特征性核心主症】(1-2个): 该证候成立时几乎必见、且最能帮助与其它证候鉴别的症状。
- 严禁列"多个证候共有的泛化症状": 如乏力在气虚/血虚/阳虚类多个证候都见、发热在多个热证都见——只有当它正是该证诊断关键时才列。
- 优先选病位/病机特异性症状: 如痰湿蕴肺选"咳嗽/咳痰"等痰湿阻肺的表现。
- 宁缺毋滥: 拿不准就不列, 绝不为凑数硬列。
- 必须逐行输出, 每行一个症状, 禁止省略、禁止写"等"。
- 【置信度】每行末尾给出 0 到 1 之间的数字, 表示你判定该边在真实临床中成立的把握(1=非常确定, 0=无把握); 未列出的候选症状即为不存在(无需输出0)。

【输出格式】每行严格为: 证候名 | 症状名 | 辨证依据(≤25字) | 置信度(0-1)
示例行: 痰湿蕴肺 | 咳嗽 | 咳嗽咳痰为痰湿蕴肺核心主症 | 0.95
只输出这些行, 不要任何解释、编号或多余文字。"""
    user = """【证候】%s
【候选症状】%s
请输出 证候名 | 症状名 | 辨证依据 | 置信度:""" % (key, syms)
    return sys_p, user


def ask_z2a(layers, key):
    ats = "、".join(norm_name(x) for x in layers["A0-治法"])
    sys_p = """你是中医肺癌辨证论治领域专家, 依据《肺癌中医诊疗指南》及中医辨证论治理论, 完成"证候->治法"的证法相符知识归纳。

【硬性规则】
- 只给【一个】标准对证治法(该证候确立后临床首选的规范治法)。
- 治法名必须一字不差取自候选治法; 若候选无对应治法, 输出"无"。
- 只依据中医理论, 不要臆造候选之外的治法名。
- 必须逐行输出, 禁止省略。
- 【置信度】每行末尾给出 0 到 1 之间的数字, 表示你判定该证-法配伍在真实临床中成立的把握(1=非常确定, 0=无把握)。

【输出格式】每行严格为: 证候名 | 治法名 | 依据(≤20字) | 置信度(0-1)
示例行: 痰湿蕴肺 | 燥湿化痰 | 痰湿蕴肺对证燥湿化痰 | 0.98
只输出这些行, 不要任何解释、编号或多余文字。"""
    user = """【证候】%s
【候选治法】%s
请输出 证候名 | 治法名 | 依据 | 置信度:""" % (key, ats)
    return sys_p, user


def ask_a2y(layers, key):
    ys = "、".join(norm_name(x) for x in layers["Y1-结局"])
    sys_p = """你是中医肺癌辨证论治领域专家, 依据《肺癌中医诊疗指南》及中医理论, 完成"治法->结局"的疗效知识归纳。

【硬性规则】
- 列出该治法在肺癌治疗中【公认能直接改善】的结局, 从候选结局中选; 每个结局一行。
- 若该治法对候选结局无直接作用, 输出"无"。
- 只依据中医理论: 补益/活血/清热等对证治法可改善症状; 化痰/降火等针对病机转归的治法可改善二诊证候。
- 宁缺毋滥, 把握不准就不列。
- 必须逐行输出, 每行一个结局, 禁止省略、禁止写"等"。
- 【置信度】每行末尾给出 0 到 1 之间的数字, 表示你判定该治法能直接改善该结局的把握(1=非常确定, 0=无把握)。

【输出格式】每行严格为: 治法名 | 结局名 | 依据(≤20字) | 置信度(0-1)
示例行: 益气养阴 | 症状改善 | 益气养阴扶正改善症状 | 0.9
只输出这些行, 不要任何解释、编号或多余文字。"""
    user = """【治法】%s
【候选结局】%s
请输出 治法名 | 结局名 | 依据 | 置信度:""" % (key, ys)
    return sys_p, user


# --------------------------------------------------------------------- 询问
def ask_one(task, key, src_ok, tgt_ok, prompt_fn, layers, base_url, model_id, raw_dir, temperature=0.0):
    log("  [%s] %s ..." % (task, key))
    sys_p, user = prompt_fn(layers, key)
    t0 = time.time()
    text = call_llm(base_url, model_id, sys_p, user, temperature)
    wt = time.time() - t0
    with open(os.path.join(raw_dir, "%s_%s.txt" % (task, key)), "w", encoding="utf-8") as f:
        f.write(text)
    edges = parse_lines(text, src_ok, tgt_ok)
    if not edges:
        log("    解析0条, 重试1次...")
        time.sleep(2)
        text = call_llm(base_url, model_id, sys_p, user, temperature)
        with open(os.path.join(raw_dir, "%s_%s.txt" % (task, key)), "w", encoding="utf-8") as f:
            f.write(text)
        edges = parse_lines(text, src_ok, tgt_ok)
    if not edges:
        fb = fallback_parse(text, key, tgt_ok)
        if fb:
            log("    主解析0条, 格式C兼容解析%d条" % len(fb))
            edges = fb
    log("    返回%d条边, wall=%.1fs" % (len(edges), wt))
    for s, t, b, c in edges:
        log("      %-8s -> %-8s | %s | conf=%s" % (s, t, b[:12], c))
    return edges


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--root", default=EXP,
                    help="方法目录的父目录; 默认 $KED_ROOT/exp（可用 KED_EXP 覆盖）")
    ap.add_argument("--layers", default=LAYERS_DEF)
    ap.add_argument("--task", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="解码温度; K=9 稳健性实验用 0.7")
    ap.add_argument("--suffix", default="",
                    help="方法目录后缀, 如 _k1 -> method_m1_k1")
    args = ap.parse_args()

    base_url, model_id, name = MODELS[args.method]

    with open(args.layers, encoding="utf-8") as f:
        layers = json.load(f)

    method_dir = os.path.join(args.root, "method_%s%s" % (args.method, args.suffix))
    raw_dir = os.path.join(method_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_csv = os.path.join(method_dir, "edges.csv")
    meta_f = os.path.join(method_dir, "meta.json")

    tasks = [args.task] if args.task else TASKS
    done = {}
    if os.path.exists(out_csv) and not args.force:
        for r in csv.DictReader(open(out_csv, encoding="utf-8-sig")):
            done.setdefault(r["combo"], set()).add(r["source"])

    defs = {
        "S0-Z0": (layers["Z0-证候"], layers["S0-症状"], ask_s2z),
        "Z0-A0": (layers["Z0-证候"], layers["A0-治法"], ask_z2a),
        "A0-Y1": (layers["A0-治法"], layers["Y1-结局"], ask_a2y),
    }

    new_rows, meta = [], {"method": args.method + args.suffix, "model": name, "ollama_id": model_id,
                          "prompt": "V0+conf", "temperature": args.temperature, "tasks": {}}
    t_start = time.time()
    no_conf = 0
    for task in tasks:
        src_nodes, tgt_nodes, prompt_fn = defs[task]
        log("=" * 60)
        log("任务 %s: %d 个源节点" % (task, len(src_nodes)))
        meta["tasks"][task] = {"n_src": len(src_nodes), "n_tgt": len(tgt_nodes)}
        for key in src_nodes:
            nkey = norm_name(key)
            if not args.force and task in done and nkey in done[task]:
                log("  [%s] %s 已有缓存, 跳过" % (task, nkey))
                continue
            edges = ask_one(task, nkey, {key}, set(tgt_nodes), prompt_fn,
                            layers, base_url, model_id, raw_dir, args.temperature)
            for s, t, b, c in edges:
                if c is None:
                    no_conf += 1
                new_rows.append({"source": s, "target": t, "directed": True,
                                 "confidence": c if c is not None else 1.0,
                                 "basis": b, "combo": task})
            time.sleep(1)

    meta["wall_s"] = round(time.time() - t_start, 1)
    meta["new_edges"] = len(new_rows)
    meta["missing_conf_defaulted_1"] = no_conf
    json.dump(meta, open(meta_f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

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
    else:
        log("无新边")

    # 汇总打印
    log("=" * 60)
    log("method_%s 完成: %d 条边 (缺失置信度按1.0计 %d 条), wall=%.0fs"
        % (args.method, len(new_rows), no_conf, meta["wall_s"]))


if __name__ == "__main__":
    main()
