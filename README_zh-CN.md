# 跨癌种图预训练聚类：蛋白质语言模型特征的鲁棒性与任务边界诊断

投稿 ***Cell Reports Methods***（Methods Track，IF ~14.3）稿件配套的代码与可复现流水线。

> 🌐 **[English README](./README.md)** | 📖 本文件为中文说明文档  
> ☝️ 点击上方「English README」切换到英文版（GitHub 默认展示）

本仓库为审稿人提供五个顶层入口：

1. **R 语言封装** `R/RunMultiCancerSSL.R` —— 五个高层动词（`build_graph / pretrain / evaluate_ppi / evaluate_patient / make_figures`）封装 Python 流水线，便于 R 语言审稿人直接调用。
2. **根目录 Makefile** —— 五个标准目标与上述五个动词一一对应：`data / preprocess / train / evaluate / figures`。
3. **10 命令 Shell 复现配方**（见下文「10 命令复现配方」）—— 从公开原始数据下载开始，端到端重建稿件中的每一张图与每一张表。
4. **Key Resources Table 产物** —— 稿件 STAR Methods 第一小节，30 行 × 9 类（Chemicals / Assays / 6 个 Deposited datasets / Cell lines / Oligos / 7 个 Software packages / 2 个 Other categories）。
5. **检查点来源登记表** —— `results/checkpoint_registry.json`（96 行 × 12 字段：checkpoint id → 文件路径 → 检查点 SHA256 12 位前缀 → 训练 run-ID → 随机种子 → 解码器结构）。

我们在由 TCGA（UCSC Xena）、STRING v12 PPI 与 ESM-2 蛋白质语言模型嵌入构建的多癌种分子网络上预训练一个异质图 Transformer（HGT），并比较三种自监督任务——NAMR（降噪重建）、CSP（PPI 链接预测）、C3L（跨癌对比学习）——单独与联合组合的表现，覆盖五个等预算随机种子 42/7/123/21/99，以及两种硬件平台（NVIDIA CUDA 与 Apple MPS）。

## 仓库结构

```
<repo root>                      ← **GitHub 投稿根目录（在此 git init）**
├── Makefile                      5 个目标：data / preprocess / train /
│                                  evaluate / figures
├── Dockerfile                    CUDA 容器（PyTorch + torch-geometric）
├── README.md                     英文 README（GitHub 默认展示）
├── README_zh-CN.md               本文件（中文说明）
├── PIPELINE.md                   端到端执行指南（按 stage 逐步）
├── requirements.txt              锁定的 Python 环境
├── project.md                    项目日志（唯一允许的中文文件；见「语言规范」）
├── CITATION.cff                  「Cite this repository」元数据（7 位作者）
├── CODE_OF_CONDUCT.md            Contributor Covenant v2.1
├── LICENSE                       MIT
├── R/RunMultiCancerSSL.R         R 封装：5 个动词（build_graph / pretrain /
│                                  evaluate_ppi / evaluate_patient / make_figures）
├── docs/                         DATA_LAYOUT / FIGURE_REVIEW / RENAME_MAP /
│                                  RENAME_MAP.v2 / REORGANIZATION_REPORT
├── figures/                      正式发表图表（300-dpi PNG + 矢量 PDF）：
│                                  Fig.1–3、Fig.S1–S5、graphical_abstract_v1
├── scripts/                      81 个流水线文件（76 Python + 1 R + 4 shell 驱动）
│   ├── config.py                 全部路径 + 协议常量（GFM_* 可覆盖）
│   ├── gfm_losses.py             损失函数 / 负采样器 / C3L InfoNCE
│   ├── figure_data.py            图表数据加载器（读取真实产物）
│   ├── figure_style.py           调色板 + rcParams（PDF+Cairo、300dpi PNG）
│   ├── 10–11_*                   数据获取（STRING v12、TCGA Xena）
│   ├── 20–39_*                   异质图构建（ESM / TCGA / 患者级）
│   ├── 30_embed_esm2             ESM-2 8m/650m 嵌入
│   ├── 40–41_*                   HGT 预训练（CSP+NAMR+C3L 三任务联合）
│   ├── 50–56_*                   一级 PPI AUC + 聚类 + 通路评估
│   ├── 60–66_*                   外部验证队列 + 药物响应
│   ├── 70–79_*                   功能富集、探针、hold-out 图
│   ├── 80–82_*                   损失聚合 + 结果汇总 + 表格
│   ├── 84–87_*                   Fig.1 布局缓存 + 图表生成
│   ├── 88–90_*                   稿件 / 数值 / 参考文献审计套件
│   ├── 93–95_*                   几何、组织-PPI 与路径归一化审计
│   ├── 100–103_*                 队列、跨种子与复现状态审计
│   ├── 120–137_*                 审稿人响应诊断与基线
│   └── run_*.sh                  4 个 shell 驱动（ablation / holdout / NAMR sweep）
├── tests/                        回归测试（3 个模块，43 项断言）
└── .gitignore                    data/ results/ models/ cloud_results/ 已排除
                                  （分别托管于 Zenodo/SWH/OSF）
```

*关于大体积二进制文件：* `data/`、`results/`、`models/` 与 `cloud_results/` 有意被 gitignore 排除；请从稿件 **Data and Code Availability** 一节引用的 Zenodo/SWH/OSF 存档中下载或迁移。

## 脚本命名

每个可执行脚本遵循 `NN_verb_object.py`；`NN` 决定该文件在其 stage 内的顺序。共享库以模块名被 import，因此不带数字前缀（Python 标识符不能以数字开头）：

| Stage | 前缀 | 文件 |
|---|---|---|
| 下载 | `10`–`11` | `10_download_string_ppi.py`、`11_download_tcga_xena.R` |
| 图 / 特征 | `20`–`39` | `30_embed_esm2.py`（`--model 8m\|650m`）、`36_build_graph_variants.py`（`--variant esm650_full\|esm20c\|esm650_20c`） |
| 预训练 | `40`–`41` | `40_pretrain.py`、`41_pretrain_single_cancer.py` |
| 评估 | `50`–`56` | `50_eval_checkpoint.py` … `56_center_check_diag.py` |
| 下游 / 外部验证 | `60`–`66` | `60_metabric_validation.py`、`63_paad_transfer_eval.py`、`64_drug_response_gdsc.py` |
| 功能分析 / 探针 | `70`–`76` | `70_go_enrichment.py`、`74_c3l_permutation_test.py` |
| Hold-out 与对照图 | `77`–`79` | `77_build_holdout_graph.py`、`79_build_control_graphs.py` |
| 结果汇总 / 表格 | `80`–`82` | `80_collect_losses.py`、`81_assemble_results.py`、`82_gen_final_tables.py` |
| 图表与图规格检查 | `84`–`87` | `84_extract_fig1_layout.py`、`85_make_figures.py`、`86_check_figure_specs.py` |
| 稿件 / 数值 / 参考文献审计 | `88`–`90` | `88_audit_manuscript.py`、`89_audit_numbers.py`、`90_audit_references.py` |
| 审稿人响应诊断 | `93`–`95`、`100`–`103`、`120`–`137` | `125_namr_denoiser_diagnostic.py`、`129_graph_smoothing_baseline.py`、`134_c3l_elimination_ablation.py` |
| 基准测试 | `122` | `122_bench_device.py` |

以 import 方式使用的共享库：`config.py`（全部路径与协议常量，可用环境变量覆盖）、`gfm_losses.py`（损失函数与负采样器）、`figure_data.py`（每一个绘图数值均从产物中读取）、`figure_style.py`（调色板、rcParams、300-dpi PNG + 矢量 PDF 导出）。

## 如何运行

所有脚本均通过 `config.py` 解析路径，因此仓库支持两种布局：

**A. 数据放在仓库之外（笔记本 / 集群 checkout 场景推荐）**

```bash
# directory holding data/, results/, models/, cloud_results/
export GFM_PROJECT_ROOT=/path/to/project
# example: evaluate the full-joint pretrained checkpoint
python scripts/50_eval_checkpoint.py \
  --ckpt results/ablation_real/full.pt --tag full
```

**B. 自包含 checkout** —— 将 `data/`、`results/`、`models/` 与 `cloud_results/` 直接放在仓库根目录下；此时 `config.PROJECT_ROOT` 即等于仓库根目录，无需设置任何环境变量。

最小端到端流程见 `PIPELINE.md`。

## 10 命令复现配方（Cell Reports Methods Methods Track）

在干净机器上依次执行以下 10 条命令，即可从公开原始数据重建稿件中的全部图表。前置条件：Python 3.11、Python venv、R 4.3+、GNU Make。

```bash
# (1) clone the repository and enter the flat submission root
git clone <repo-url> gfm-crm && cd gfm-crm

# (2) create + activate the pinned Python virtual environment
python3 -m venv .venv && source .venv/bin/activate

# (3) upgrade pip and install the pinned PyTorch / PyG / ESM2 stack
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# (4) verify the R wrapper (5 verbs; needs R 4.3+, base-R only)
Rscript R/RunMultiCancerSSL.R

# (5) TARGET 1 — download every raw public input (STRING v12, TCGA Xena, ...)
make data

# (6) TARGET 2 — ESM-2 embeddings + all graph variants
make preprocess

# (7) TARGET 3 — equal-budget pretraining (5 seeds × 500 epochs, 5 arms)
make train

# (8) TARGET 4 — downstream evaluation (PPI / cluster / pathway / METABRIC / PAAD)
make evaluate

# (9) TARGET 5 — regenerate every manuscript figure (PDF + 300-dpi PNG)
make figures

# (10) optional (non-required for results): run the 43 regression tests
python tests/test_gfm_losses.py && python tests/test_new_scripts.py && python tests/test_figure_data.py
```

可用于路径覆盖的环境变量：`GFM_PROJECT_ROOT`、`GFM_DATA_RAW`、`GFM_DATA_PROC`、`GFM_MODEL_DIR`、`GFM_RESULT_DIR`、`GFM_CLOUD_DIR`、`GFM_FIG_DIR`、`GFM_STRING_DIR`、`GFM_SEED`、`GFM_PYTHON`（详见 `scripts/config.py`）。

## 数据与代码可用性

全部预处理、预训练、评估与绘图代码均在本仓库内。训练好的检查点、派生嵌入与公开原始数据单独归档（见稿件的数据可用性声明）；若缺少这些归档，流水线会从公开源重新构建数据，但不会复现论文中的精确数值。

## 图表

```bash
# cache the Fig.1(a) network layout precomputed from the real heterograph
python scripts/84_extract_fig1_layout.py
# regenerate fig1 / fig2 / fig3 / S1–S5 + GraphicalAbstract from real artifacts
python scripts/85_make_figures.py
# pick a subset:
python scripts/85_make_figures.py --only fig1 fig3
```

每张图均以 300-dpi PNG 加含活字的矢量 PDF 写入 `GFM_FIG_DIR`（默认 `<PROJECT_ROOT>/manuscript`）。绘图代码中没有硬编码数值：所有数值均从 `results/ablation_real/eval_summary.txt`、检查点损失历史，以及 `docs/FIGURE_REVIEW.md` 列出的来源 JSON 文件中读取。`tests/test_figure_data.py` 会在任何重跑或审计之后断言绘图数值仍与这些产物一致。

## 测试

```bash
# 9 loss / sampler unit tests
python tests/test_gfm_losses.py
# 5 statistics / new-script helper tests
python tests/test_new_scripts.py
# 29 checks: every plotted value matches the referenced artifact file
python tests/test_figure_data.py
```

## 语言规范

- **代码、注释、终端输出与图表：** 仅使用英文。对 `scripts/`、`tests/` 与 `R/` 执行 CJK 扫描结果为空。
- **`project.md`：** 中文——本项目唯一的中文权威日志（Rule 13）。
- **`README.md`、`PIPELINE.md` 与 `docs/`：** 英文。
- **`README_zh-CN.md`：** 中文——本文件的英文版对照。本仓库中仅 `project.md` 与本文件含 CJK 字符。

## 引用

（稿件 DOI / 引用信息占位符——投稿时填入）

---

> 🌐 [返回 **English README**](./README.md) | 最后更新：2026-10-17
