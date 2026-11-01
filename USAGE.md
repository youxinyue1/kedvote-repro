# 代码使用说明

本文件说明如何**逐步运行** `kedvote-repro` 中的代码、每步产生什么、以及如何判断复现是否成功。
仓库概览与方法背景见 [`README.md`](README.md)。

| 项目 | 内容 |
|---|---|
| 对应论文 | Heterogeneous LLM Ensembles with Confidence Voting Reconstruct TCM Lung Cancer Causal Graphs（KED-Vote） |
| 可复现范围 | 主文 Table 1、4、5、6、7、8、9、10、11 与附录 S1–S5 的**全部数值** |
| 不可复现部分 | 配图（作图脚本未随仓库发布）、文稿排版 |
| 两条路线 | A 完整复现（需 GPU + 12 个开源模型，约 2 GPU 小时）；B 只复现评测与统计（零 GPU） |

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux（远程实验）或 Windows / macOS（仅评测统计路线） |
| Python | ≥ 3.9 |
| Python 依赖 | `scipy ≥ 1.10`（其余全部使用标准库） |
| 推理服务 | Ollama ≥ 0.3（默认 `http://127.0.0.1:11434`，OpenAI 兼容端点）或 llama.cpp 的 `llama-server` |
| GPU | 单卡 32 GB 可跑全部模型；Yi-1.5-34B（Q4_0）需 ≥ 24 GB |

安装：

```bash
git clone <your-repo-url> kedvote-repro
cd kedvote-repro
pip install -r requirements.txt
```

---

## 2. 文件清单

### `scripts/`（复现代码）

| 脚本 | 作用 | 需要 GPU | 主要输出 |
|---|---|---|---|
| `theory_dag_exp.py` | 单模型逐节点询问，输出带 0–1 置信度的边 | 是 | `exp/method_mN/{edges.csv, raw/, meta.json}` |
| `theory_dag_m5.py` | 提示组件消融（B 轴 6 组） | 是 | `exp/method_m5*/` |
| `ked_vote.py` | **KED-Vote 主体**：C 轴 4 专家团 × D 轴 4 投票策略 = 16 组 | 否（读已有边） | `exp/method_m12_cX_dY/` |
| `pc_m7.py` | 数据驱动 PC 算法对照（卡方/Fisher + 条件独立 + 层序定向） | 否 | `exp/method_m7/edges.csv` |
| `eval_comparison.py` | 39 个配置统一评测（AUC/AUPR/P/R/F1） | 否 | `exp/eval/comparison_summary.csv`、`comparison_report.md` |
| `per_method_results.py` | 逐方法 282 行明细表 | 否 | `results/per_method/*.csv` |
| `per_instance_results.py` | 逐候选边宽表/长表 | 否 | `results/per_edge_predictions_{wide,long}.csv` |
| `k9_variance.py` | K=9 重复解码稳健性 | 否 | `exp/eval/k9_variance.{csv,txt}` |
| `loeo_sensitivity.py` | 17 折逐边留一敏感性 | 否 | `results/loeo_sensitivity.csv`、`loeo_summary.txt` |
| `selfconsistency_baseline.py` | 预算匹配的自洽性对照（SC-mean / SC-majority / SC-K4） | 否 | `results/eval/selfconsistency_summary.csv` |
| `stats_significance.py` | DeLong / bootstrap / McNemar / BH-FDR | 否 | `results/significance_tests.txt` |
| `panel_analysis.py` | 11×11 单体模型 Spearman 相关性矩阵 | 否 | `results/eval/panel_correlation.txt` |
| `preflight.py` | 推理链路连通性预检 | 否 | 终端输出 |

### `run/`（远程服务器批量脚本）

`run_all_v32.sh`（m3→m2→m1→m4b→m4）、`run_all_v4.sh`（m9c/m9b/m8b/m9/m10）、`run_m5.sh`、`run_m11.sh`、`run_k9.sh`（K=9）、`run_all_v0.sh`、`servers.sh`（服务启停）。
脚本内 `/data/...` 为原服务器路径，已改为 `${CODE_DIR:-/data/exp}`、`${EXP_ROOT:-/data/result/exp}`，**请按实际环境覆盖或改写**。

---

## 3. 数据准备

| 文件 | 内容 | 用途 |
|---|---|---|
| `data/node_layers.json` | 4 层标准节点集：S0 症状 / Z0 证候 / A0 治法 / Y1 结局 | 所有步骤 |
| `data/ground_truth_aligned.csv` | 17 条金标准边（节点名已对齐标准术语表） | 所有评测 |
| `data/extended_data.csv` | D1 病例二值矩阵（391 例 × 90 节点） | 仅 PC 算法对照（步骤 4）；不公开可删除 |

**替换为自己的数据时要满足**：节点名须带层前缀（`症状::`、`证候::`、`治法::`、`结局::`），
金标准边与候选边都取自层间有序对（S0→Z0、S0→Y1、Z0→A0、Z0→Y1、A0→Y1），
未生成的边记 `confidence = 0`。

---

## 4. 路径与端点配置

三个环境变量（脚本文件头统一读取，未设置时用仓库内相对路径）：

```bash
export KED_ROOT=/path/to/kedvote-repro   # 结果输出根，默认仓库根
export KED_DATA=$KED_ROOT/data           # 默认 $KED_ROOT/data
export KED_EXP=/path/to/exp              # 各 method_* 目录的父目录，默认 $KED_ROOT/exp
```

推理端点与模型注册名在 `scripts/theory_dag_exp.py` 顶部：

```python
OLLAMA = "http://127.0.0.1:11434"        # 改这里可指向其他主机/端口
MODEL_CFG = {"m1": (OLLAMA, "qwen2.5:14b", ...), ...}   # Ollama 中注册的模型名
```

12 个模型 ID：m1 Qwen2.5-14B、m2 Qwen2.5-7B、m3 Qwen2.5-3B、m4 扁仓-14B、m4b 医承有道-7B、
m8 Baichuan2-13B、m8b Baichuan2-7B、m9 DeepSeek-R1-Distill-14B、m9b 7B、m9c 1.5B、m10 GLM-4-9B、m11 Yi-1.5-34B。

---

## 5. 路线 A：完整复现

> 约定：以下均在 `kedvote-repro/` 下执行，且已设置第 4 节的环境变量。

### 步骤 0 — 启动推理服务

```bash
ollama serve                       # 或 bash run/servers.sh start m1
python scripts/preflight.py        # 连通性预检，通过再往下跑
```

### 步骤 1 — 12 个单模型推理 → 主文 Table 4

```bash
for m in m1 m2 m3 m4 m4b m8 m8b m9 m9b m9c m10 m11; do
  python scripts/theory_dag_exp.py --method $m \
      --root $KED_EXP --layers data/node_layers.json --force
done
```

- 产物：`$KED_EXP/method_$m/edges.csv`（六列：`source,target,directed,confidence,basis,combo`）+ `raw/` 原始回答 + `meta.json`
- 断点续跑：已存在 `edges.csv` 时默认跳过，`--force` 强制重跑
- 建议顺序 3B → 7B → 14B → 34B；显存不足可跳过 m11

### 步骤 2 — 提示组件消融 → 主文 Table 5

```bash
for m in m5 m5_role_off m5_cot_off m5_lex_off m5_fewshot_off m5_conf_off; do
  python scripts/theory_dag_m5.py --method $m --root $KED_EXP --layers data/node_layers.json
done
```

### 步骤 3 — KED-Vote 集成 → 主文 Table 6

```bash
python scripts/ked_vote.py
```

读取各成员 `edges.csv`，生成 16 组：`method_m12_c{1,2,3,4}_d{1,2,3,4}`。
主配置为 **c3 × d3**（通用+领域混合 4 员 · 置信度均值）。

### 步骤 4 — PC 算法对照 → 主文 Table 9

```bash
python scripts/pc_m7.py                                   # D1
python scripts/pc_m7.py --tag m7d2 --label D2             # 跨源复现（需自备 D2 矩阵）
```

### 步骤 5 — 统一评测 → 主文 Table 4/5/6/7、附录 S1

```bash
python scripts/eval_comparison.py            # → exp/eval/comparison_summary.csv（39 行）
python scripts/per_method_results.py         # → results/per_method/*.csv
python scripts/per_instance_results.py       # → results/per_edge_predictions_{wide,long}.csv
```

### 步骤 6 — 稳健性与统计检验

```bash
# 6a K=9 重复解码（需 GPU）
for k in 1 2 3 4 5 6 7 8 9; do
  python scripts/theory_dag_exp.py --method m1 --temperature 0.7 --suffix _k$k --force
done
python scripts/k9_variance.py                # → 主文 Table 10、附录 S2

# 6b 把边目录挂到 results/edges/（panel / selfconsistency 从这里读）
mkdir -p results/edges
cp -r $KED_EXP/method_m* results/edges/

python scripts/loeo_sensitivity.py           # → 主文 Table 11、附录 S3（依赖 6a 前的 wide 表）
python scripts/selfconsistency_baseline.py   # → 附录 S4
python scripts/stats_significance.py         # → 主文 Table 8
python scripts/panel_analysis.py             # → 附录 S5
```

> `panel_analysis.py` 与 `selfconsistency_baseline.py` 读取 `results/edges/method_<id>/edges.csv`，
> 缺失会打印 `skip (no local edges): mX`，此时用上面的 `cp -r` 补齐即可。
> `loeo_sensitivity.py` 依赖 `results/per_edge_predictions_wide.csv`，须先跑 `per_instance_results.py`。

---

## 6. 路线 B：只复现评测与统计（零 GPU）

若已有各方法 `edges.csv`（例如从论文补充材料或本机归档取得），按如下摆放后直接从**步骤 5** 开始：

```
exp/method_m1/edges.csv  …  exp/method_m12_c3_d3/edges.csv
```

随后执行步骤 5、6b（跳过 6a 的 GPU 推理，K=9 相关输出不可用）。
评测口径完全由 `data/` 中的金标准与节点集决定，不依赖任何模型权重。

---

## 7. 参数速查

| 脚本 | 参数 | 说明 |
|---|---|---|
| `theory_dag_exp.py` | `--method` | 必填，取值见第 4 节 12 个模型 ID |
| | `--root` / `--layers` | 产物父目录 / 节点集路径 |
| | `--task` | 只跑某一层对，如 `S0-Z0` |
| | `--temperature` | 默认 0.0（确定性）；K=9 实验用 0.7 |
| | `--suffix` | 产物目录后缀，`_k1` → `method_m1_k1` |
| | `--force` | 已存在产物时强制重跑 |
| `theory_dag_m5.py` | `--method` | `m5` 及 5 个 `_off` 变体 |
| `pc_m7.py` | `--data` / `--tag` / `--label` | 病例矩阵路径 / 产物目录名 / 数据集标签 |
| `ked_vote.py` | 无 CLI | 组内常量 `C_AXIS`（c1–c4）、`D_AXIS`（d1–d4） |

---

## 8. 关键输出字段

- `edges.csv`：`source, target, directed, confidence, basis, combo`
- `comparison_summary.csv`（39 行，一行一配置）：
  `method, model, space, generated_edges, tp, fp, fn, precision, recall_17, recall_15, f1, auc, aupr`
- `significance_tests.txt`：比较对、指标、差值 Δ、检验方法（DeLong / bootstrap / McNemar）、p 值、BH-FDR 后结论
- `results/eval/panel_correlation.txt`：11×11 Spearman ρ 矩阵（附录 S5 数值来源）

评测口径：候选空间 282 条层间有序边；金标准 17 条；AUC 用平均秩、AUPR 用块积分 AP；
`recall_17` 为全金标准口径，`recall_15` 为理论层负责口径。

---

## 9. 复现成功的判据

跑完步骤 5 后，`comparison_summary.csv` 中以下配置应与论文一致（T=0 确定性解码，应逐位相同）：

| 配置 | 论文值 |
|---|---|
| KED-Vote c3d3（主方法） | Edges 140，TP 15，P 0.107，R15 1.000，F1 0.194，**AUC 0.854，AUPR 0.398** |
| KED-Vote c3d1（交集） | Edges 36，P 0.333，F1 0.471，AUC 0.818，AUPR 0.374 |
| 最佳单模型 m1（Qwen2.5-14B） | Edges 89，TP 14，P 0.157，R15 0.933，F1 0.269，**AUC 0.832，AUPR 0.293** |
| 指标冠军（Table 7） | AUC 与 AUPR 冠军均为 KED-Vote mean (c3d3)：0.854 / 0.398 |
| KED-Vote vs m1，AUPR（Table 8） | Δ = +0.106，bootstrap p = 0.014，显著 |

带随机性的结果（temperature 0.7 的 K=9 与自洽性）允许小幅波动：

| 结果 | 论文值 | 允许偏差 |
|---|---|---|
| m1 九次解码 AUC | 0.803 ± 0.020 | ± 0.02 |
| m1 九次解码 AUPR | 0.260 ± 0.013 | ± 0.02 |
| SC-mean (K=9) | AUPR 0.2976，AUC 0.8367 | ± 0.02 |
| LOEO 结论保留率 | 5 条预设结论 17/17 折全部保持 | 应一致 |

若 `m1` 的 AUC 明显偏离 0.832，优先检查：① 金标准与节点集是否用的 `data/` 下的对齐版；
② 模型注册名与量化版本（Q4_K_M / Q8_0）是否与论文一致；③ 温度是否为 0.0。

---

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| `preflight.py` 预检失败 | 确认 Ollama 在跑且 `127.0.0.1:11434` 可达；改 `OLLAMA` 常量 |
| 某模型跑批中断 | 直接重跑同一命令（已有产物会跳过）；需覆盖时加 `--force` |
| `skip (no local edges): mX` | `panel_analysis.py` / `selfconsistency_baseline.py` 需要 `results/edges/method_*`，见步骤 6b |
| `loeo_sensitivity.py` 报文件不存在 | 先跑 `per_instance_results.py` 生成 `per_edge_predictions_wide.csv` |
| 中文乱码 | 所有 CSV 以 `utf-8-sig` 写入与读取，脚本已处理；勿用 Excel 另存为 GBK |
| 配置数不是 39 | 应含 12 单模型 + 7 消融行（V0/V2）+ 16 KED-Vote + 2 legacy（m6/m6b）+ 3 PC（m7/m7d2/m7d2_391） |
| 显存不足 | 按 3B→7B→14B 顺序跑；34B（m11）可跳过，不影响主方法结论 |
| 想只验证统计口径 | 走路线 B，零 GPU 即可复现 Table 7/8/11 与附录 S1/S3/S5 |

---

## 11. 引用

## 12. 许可证

本仓库代码以 **MIT License** 发布（见 [`LICENSE`](LICENSE)），允许自由使用、修改与再分发（含商用），
须保留版权声明。`data/extended_data.csv`（391 例去标识化病例矩阵）同样随仓库提供，如用于衍生研究请注明数据来源。

论文 Declarations 中的 "Code availability" 请填写：

```
Repository: <your-repo-url>    License: MIT
```

若使用本代码，请引用对应论文。
