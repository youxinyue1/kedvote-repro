# KED-Vote 复现代码

知识增强异构专家投票（KED-Vote）用于中医肺癌因果图重建的**实验复现代码**。
只保留跑实验与算指标所必需的代码和数据；生成论文文稿（.docx）、配图、汇报 PPT 的脚本不在本仓库内。

> **逐步运行方法、参数说明与复现判据见 [`USAGE.md`](USAGE.md)。** 本文件只作仓库概览。
>
> 许可证：**MIT**（见 [`LICENSE`](LICENSE)）。

## 目录结构

```
kedvote-repro/
├── data/                          输入数据与金标准
│   ├── node_layers.json           4 层标准节点集（S0 症状 / Z0 证候 / A0 治法 / Y1 结局）
│   ├── ground_truth_aligned.csv   17 条金标准边（节点名已对齐标准术语表）
│   └── extended_data.csv          D1 病例二值矩阵（391 例 × 90 节点），仅 PC 算法对照需要
├── scripts/                       复现代码（按流水线顺序执行）
└── run/                           远程服务器批量运行脚本（Bash）
```

## 环境

- Python ≥ 3.9，仅需 `scipy`（其余全用标准库）：

```bash
pip install -r requirements.txt
```

- 推理侧：Ollama（脚本默认 `http://127.0.0.1:11434/v1/chat/completions`，OpenAI 兼容端点）或 llama.cpp 的 `llama-server`。
- 模型：12 个开源模型（Qwen2.5-14B/7B/3B、DeepSeek-R1-Distill-14B/7B/1.5B、GLM-4-9B、Yi-1.5-34B、Baichuan2-13B/7B，以及扁仓、医承有道等中医领域模型）。原文全部实验约 2 GPU 小时（V100-SXM2-32GB）。

## 路径配置

所有脚本在文件头统一读取以下环境变量，未设置时使用仓库内相对路径：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `KED_ROOT` | 仓库根目录 | 结果输出根 |
| `KED_DATA` | `$KED_ROOT/data` | 节点集、金标准、病例矩阵 |
| `KED_EXP` | `$KED_ROOT/exp` | 各方法产物目录 `method_*/` 的父目录 |

```bash
export KED_ROOT=/path/to/kedvote-repro
export KED_EXP=/path/to/exp      # 可选，默认 $KED_ROOT/exp
```

`run/` 下的 Bash 脚本另有 `CODE_DIR`（默认 `/data/exp`）与 `EXP_ROOT`（默认 `/data/result/exp`）两个变量，
**这两个默认值是原远程服务器的路径，需按实际环境改写或直接传参覆盖**。
`run/servers.sh` 内的 `/data/models/...`、`/data/llama/bin` 同为服务器路径，需自行替换。

## 复现流水线

| 步骤 | 命令 | 产物 | 对应论文 |
|---|---|---|---|
| 0 起服务 | `bash run/servers.sh start m1` | Ollama / llama-server 端点 | — |
| 1 单模型推理 | `python scripts/theory_dag_exp.py --method m1 --root $KED_EXP --layers data/node_layers.json` | `exp/method_mN/{edges.csv, raw/, meta.json}` | Table 1、Table 4 |
| 2 提示组件消融（B 轴） | `python scripts/theory_dag_m5.py --method m5`（另跑 `m5_role_off` / `m5_cot_off` / `m5_lex_off` / `m5_fewshot_off` / `m5_conf_off`） | `exp/method_m5*/` | Table 5 |
| 3 KED-Vote 集成（C×D 轴） | `python scripts/ked_vote.py`（读完 `method_m1/m9/m4/...` 后生成 16 组） | `exp/method_m12_cX_dY/` | Table 6 |
| 4 PC 算法对照 | `python scripts/pc_m7.py` | `exp/method_m7/edges.csv`、`dropped_cond_indep.csv` | Table 9 |
| 5 统一评测 | `python scripts/eval_comparison.py` | `exp/eval/comparison_summary.csv`、`comparison_report.md` | Table 4/5/6/7、附录 Table S1.1 |
| 5b 逐方法/逐边明细 | `python scripts/per_method_results.py`、`python scripts/per_instance_results.py` | `results/per_method/*.csv`、`results/per_edge_predictions_{wide,long}.csv` | 附录 S1；wide 表供 LOEO 使用 |
| 6a 重复解码稳健性 | `python scripts/k9_variance.py`（需先跑 `run/run_k9.sh` 产 K=9 归档） | `exp/eval/k9_variance.{csv,txt}` | Table 10、附录 S2 |
| 6b 逐边留一敏感性 | `python scripts/loeo_sensitivity.py` | `results/loeo_sensitivity.csv`、`loeo_summary.txt` | Table 11、附录 S3 |
| 6c 自洽性对照 | `python scripts/selfconsistency_baseline.py` | `results/eval/selfconsistency_summary.csv`、`selfconsistency_k4_bootstrap.txt` | 附录 S4 |
| 6d 显著性检验 | `python scripts/stats_significance.py` | `significance_tests.txt` | Table 8 |
| 6e 面板误差相关性 | `python scripts/panel_analysis.py` | `results/eval/panel_correlation.txt` | 附录 S5（11×11 Spearman 矩阵） |

`scripts/preflight.py` 是推理链路连通性预检，跑大批量前建议先执行。

## 评测口径

- 候选空间：标准节点集层间有序对共 **282** 条（S0→Z0、S0→Y1、Z0→A0、Z0→Y1、A0→Y1）；未生成的边记 conf = 0。
- 金标准：**17** 条边。
- AUC 采用平均秩算法，AUPR 采用块积分 AP；同时给出 precision / F1 与双口径召回 **R17**（全金标准）和 **R15**（理论层负责口径）。
- 统计检验：DeLong 配对 AUC 检验、bootstrap 置信区间与配对差、McNemar（top-50 截断）、Benjamini–Hochberg FDR 校正。

## 本仓库不含的内容

- 生成论文文稿：`.server_bridge/_fmt_*.py`、`scripts_tmp/make_supplement.py`、`stage1/2/3_*`、`文稿/manuscript_build.js`
- 配图：`make_figures.py` / `make_figures_v2.py` / `make_figures_en.py` / `fig_style.py` / `fig5_v2.py` 等
- 汇报 PPT：`文稿/ked_v3.js`、`文稿/kedvote_fig_ppt.js`
- 远程服务器运维：`setup_models*.sh`、`fix_*.sh`、`probe*.sh`、`pull_exp.py`、`download_models.py`、`ssh_bridge.py`
- 含凭据的 `server_config.json`（**不入库**；评测脚本已改为读取 `KED_EXP` 环境变量）

## 数据说明

`data/extended_data.csv` 是 391 例中医肺癌验案的二值节点矩阵，仅 PC 算法对照（步骤 4）需要。
若不打算公开该临床数据，删除此文件并跳过步骤 4 即可，其余结果不受影响。
