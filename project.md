# Project Documentation — Cross-cancer Graph-Pretrained Clustering (GFM)

> **文档唯一权威记录 | Single Source of Truth**
>
> 文档版本: v2.0.4
> 创建日期: 2026-09-26
> 最后更新: 2026-10-26
> 项目负责人: Tao Zhu
> 关键词: `graph-pretraining`, `HGT`, `cross-cancer`, `NAMR`, `CSP`, `C3L`, `ESM-2`, `TCGA`, `STRING-PPI`, `Cell Reports Methods`

---

## 目录索引

1. [课题目的](#1-课题目的)
2. [详细实验路线](#2-详细实验路线)
3. [实验结果](#3-实验结果)
4. [图表文件](#4-图表文件)
5. [研究计划完成情况](#5-研究计划完成情况)
6. [变更记录](#6-变更记录)
7. [版本管理规范](#7-版本管理规范)
8. [质量保障措施](#8-质量保障措施)

> **唯一数字底账**: 本文档引用的所有数值均源自 `results/freezer/master_numbers_20261009.csv`。
> 任何数值变更必须先更新该底账，再回填本文档、正文与补充材料。

---

## 1. 课题目的

### 1.1 研究背景 | Research Background

异质图 Transformer (Heterogeneous Graph Transformer, HGT) 在多模态生物网络建模中展现了强大能力，但跨癌种图预训练的**鲁棒性**与**任务边界**尚未得到系统性评估。现有研究存在以下关键空白：

- **任务耦合机制不明**: NAMR（降噪重建）、CSP（PPI 链接预测）、C3L（跨癌对比学习）三种自监督任务的协同/干扰机制缺乏量化分析。
- **特征来源边界不清**: ESM-2 蛋白质语言模型嵌入与原始组学特征的贡献边界未通过严格对照实验验证。
- **可复现性存疑**: 不同硬件平台（CUDA vs Apple MPS）、不同随机种子下的结果稳定性缺乏系统报告。

本研究基于 **TCGA (UCSC Xena) pan-cancer 表达谱 + STRING v12 PPI 网络 + ESM-2 蛋白语言模型嵌入** 构建异质分子网络，系统性地评估三任务联合预训练策略的有效性与边界。

### 1.2 研究目标 | Research Objectives

| 编号 | 目标内容 | 验证方式 |
|---|---|---|
| O1 | 量化 NAMR / CSP / C3L 三种预训练任务单独与联合的性能差异 | 消融实验（4 实验组 × 5 种子 × 2 平台） |
| O2 | 解析 ESM-2 嵌入在图预训练中的作用边界 | 特征消融（raw / ESM-2 8M / ESM-2 650M）+ 等体积对照图 |
| O3 | 评估跨癌种迁移能力（PAAD hold-out）与患者级预测性能 | 外部验证集（METABRIC、PAAD）、患者级 P-NAMR 协议 |
| O4 | 提供可复现、可审计的完整分析管线 | 开源代码 + 版本化数据契约 + 自动回归测试（43 项断言） |

### 1.3 预期成果 | Expected Deliverables

- 1 篇投稿至 *Cell Reports Methods* (Methods Track) 的研究论文。
- 正式发表图表：3 幅主图（Fig. 1–3）+ 5 幅补充图（Fig. S1–S5）+ 图形摘要（Graphical Abstract），均提供 PDF（矢量）与 PNG（300 dpi）双版本。
- 完整可复现脚本流水线（`scripts/`）：覆盖下载 → 构图 → 预训练 → 评估 → 出图全流程。
- 3 套回归测试（损失正确性、统计辅助函数、图表-数据一致性），共 43 项断言。

---

## 2. 详细实验路线

### 2.1 总体技术路线 | Overall Pipeline

```
Stage 1: Download          Stage 2: Graph & Features     Stage 3: Pretraining
  STRING v12 (9606)          ESM-2 Embedding (8M/650M)      Arms: full / namr_only /
  TCGA UCSC Xena       ──▶   Heterogeneous Graphs      ──▶          csp_only / csp_namr
  GDC (alternative)           C3L Targets / Patient Edges    Seeds: 42/7/123/21/99
                                                             Platforms: CUDA / Apple MPS
        │                            │                              │
        ▼                            ▼                              ▼
Stage 4: Evaluation        Stage 5: External Valid.     Stage 6: Functional Analysis
  PPI Link Prediction (AUC)  METABRIC ER (AUROC)            GO Enrichment (per cluster)
  Few-shot Clustering (Acc)  PAAD Hold-out (Spearman ρ)     KEGG Pathway Membership
  Embedding Diagnostics      Drug Response GDSC2 (R²)       C3L Probe / Permutation Test
  Isotropy / Center Check    GTEx Tissue-specific PPI      Set-fingerprint Mechanism
        │                            │                              │
        ▼                            ▼                              ▼
Stage 7: Assembly & Figures      Stage 8: Benchmark & Audits
  Loss Summaries / Final Tables     Device / BLAS Benchmark (CUDA vs MPS)
  Figure Layouts (Fig.1a JSON)      Manuscript / Number / Reference Audit
  Manuscript Figures (PDF + PNG)    Checkpoint Integrity Audit
```

### 2.2 Stage 1 — Download public data | 公共数据下载

**阶段目标**: 获取 STRING v12 PPI 网络、TCGA pan-cancer 表达谱与临床数据，验证数据完整性。

| 步骤 | 实验操作 | 脚本 | 输出产物 |
|---|---|---|---|
| 1.1 | 下载 STRING v12 Homo sapiens (9606) 蛋白信息、别名与链接数据 | `scripts/10_download_string_ppi.py` | `data/raw/string/9606.protein.*.txt` |
| 1.2 | 下载 TCGA GDC 表达与临床数据（GDC 备选路径） | `scripts/11_download_tcga_xena.R` | `data/raw/tcga/{expression,clinical}/*.rds` |
| 1.3 | 准备 UCSC Xena pan-cancer 导出矩阵作为主数据源 | （外部数据导入） | `GFM_XENA_DIR` 路径下的 TPM 矩阵 |

⚠ **注意**: 本文使用 UCSC Xena pan-cancer 导出作为主 TCGA 数据源（见 `docs/DATA_LAYOUT.md` 的 `GFM_XENA_DIR` 配置）；`21_build_graph_xena_large.py` 在该变量未设置时会以明确报错退出。

### 2.3 Stage 2 — Embeddings, graphs, targets | 嵌入、图谱与目标构建

**阶段目标**: 生成 ESM-2 基因嵌入，构建异质基因-癌症-患者图，预计算 C3L 对比学习目标。

| 步骤 | 实验操作 | 脚本 | 输出产物 |
|---|---|---|---|
| 2.1 | 使用 ESM-2 (8M / 650M) 编码所有基因的蛋白序列 | `scripts/30_embed_esm2.py` | `esm2_gene_embeddings.pt` (320-dim)<br>`esm2_650m_gene_embeddings.pt` (1,280-dim) |
| 2.2 | 构建 5,000 基因 × 21 癌症异质图 (ESM-2 8M) | `scripts/20_build_graph_esm.py` | `xena_graph_esm.pt` |
| 2.3 | 构建 13,881 基因大规模异质图 | `scripts/21_build_graph_xena_large.py` | `xena_graph_large.pt` |
| 2.4 | 预计算 C3L 跨癌对比学习目标矩阵 | `scripts/32_precompute_c3l_targets.py` | `gene_cancer_targets*.pt` |
| 2.5 | 构建患者级异质图（基因-患者边 + PAAD 子集） | `scripts/33_*`, `scripts/34_*`, `scripts/35_*` | `xena_graph_patient_esm.pt`<br>`xena_graph_patient_paad.pt` |
| 2.6 | 构建图变体（650M 全图 / 20-cancer hold-out / 联合变体） | `scripts/36_build_graph_variants.py` | `xena_graph_esm650_large.pt`, `*_20c.pt` |
| 2.7 | 构建等体积随机边对照图 + 重连边对照图 | `scripts/39_build_random_edge_controls.py` | `rewired` / `eqvol` 对照图对象 |
| 2.8 | CI 测试用小型玩具图 | `scripts/22_build_test_graph_ci.py` | `test_graph.pt`（仅 CI，不用于分析） |

### 2.4 Stage 3 — Pretraining | 预训练

**阶段目标**: 按消融协议训练 HGT 模型，固定 5 个随机种子，覆盖 CUDA 与 Apple MPS 双平台。所有训练为 full-batch，500 epochs。

| 步骤 | 实验操作 | 脚本 | 参数配置 |
|---|---|---|---|
| 3.1 | **Full 联合预训练** (NAMR + CSP + C3L) | `scripts/40_pretrain.py` | `--epochs 500 --seed {42,7,123,21,99}` |
| 3.2 | **NAMR-only 消融** | `scripts/40_pretrain.py` | `--csp_weight 0 --c3l_weight 0` |
| 3.3 | **CSP-only 消融** | `scripts/40_pretrain.py` | `--namr_weight 0 --c3l_weight 0` |
| 3.4 | **CSP+NAMR 消融**（无 C3L） | `scripts/40_pretrain.py` | `--c3l_weight 0` |
| 3.5 | Single-cancer BRCA 等预算对照 | `scripts/41_pretrain_single_cancer.py` | `--cancer BRCA --epochs 500` |
| 3.6 | 650M 特征 × 20-cancer 等预算训练 | `scripts/40_pretrain.py` + graph variants | `--variant esm650_20c` 等 |
| 3.7 | 患者级 P-NAMR 预训练 | `scripts/62_patient_pretrain_eval.py` | `--epochs 500` |
| 3.8 | 标准化参数扫描 (NAMR-only) | `run_namr_sweep_parallel.sh` / `run_namr_sweep_local.sh` | `namr_sweep/configs.txt` |

每个 checkpoint 保存完整的 per-epoch `loss_history`，作为 Fig. 2 / S4 的数值来源。

### 2.5 Stage 4 — Evaluation | 核心评估

**阶段目标**: 评估 PPI 链接预测 (AUC) 与 few-shot 聚类准确率 (Acc)，运行嵌入空间诊断。

| 步骤 | 实验操作 | 脚本 | 评估指标 |
|---|---|---|---|
| 4.1 | 标准 checkpoint 下游评估 | `scripts/50_eval_checkpoint.py` | 2,000 pos/neg PPI pairs AUC + 30 repeats 聚类 Acc |
| 4.2 | 嵌入空间可视化与诊断 | `scripts/51_eval_gene_embeddings.py` | 聚类轮廓系数等 |
| 4.3 | Single-cancer BRCA 控制组评估 | `scripts/52_eval_single_cancer.py` | 与 3.5 对照 |
| 4.4 | Raw ESM-2 (8M / 650M) 基线评估 | `scripts/53_*`, `scripts/54_*` | 同等协议 PPI AUC / Acc |
| 4.5 | 监督式 PPI 链接预测（文献可比协议） | `scripts/55_eval_supervised_ppi.py` | 标准边预测设置 AUC |
| 4.6 | 嵌入中心 / 各向同性诊断 | `scripts/56_center_check_diag.py` | 中心偏移量 / 各向异性比 |
| 4.7 | 全消融评估批量执行 | `run_ablation_evals.sh` | 输出至 `results/ablation_real/eval_summary.txt` |

### 2.6 Stage 5 — Downstream and external validation | 下游与外部验证

**阶段目标**: 在独立数据集与任务上验证预训练特征的迁移能力与边界。

| 步骤 | 实验操作 | 脚本 | 验证集 / 任务 |
|---|---|---|---|
| 5.1 | METABRIC ER 状态分类迁移（独立乳腺癌队列） | `scripts/60_metabric_validation.py` | per-split AUROC 分布数组 |
| 5.2 | 患者级 (P-NAMR) 预测性能 | `scripts/61_patient_eval.py` | 患者-基因二部图预测 |
| 5.3 | 患者级学习曲线分析 | `scripts/62_patient_pretrain_eval.py` | 训练 epoch vs 患者 AUC |
| 5.4 | PAAD 表达谱迁移（20-cancer hold-out 设计） | `scripts/63_paad_transfer_eval.py` | Spearman ρ 数组 |
| 5.5 | GDSC2 药物反应预测（负面探针实验） | `scripts/64_*`, `scripts/65_*` | 药物 IC50 R²（GFM vs ESM vs PCA vs Random） |
| 5.6 | GTEx 组织特异性 PPI 边损失分析 | `scripts/66_tissue_specific_ppi.py` | 组织过滤后 PPI 边保留比例 |

### 2.7 Stage 6 — Functional analysis and probes | 功能分析与机制探针

**阶段目标**: 通过 GO/KEGG 富集、置换检验、机制探针实验验证预训练特征的生物学可解释性。

| 步骤 | 实验操作 | 脚本 | 分析方法 |
|---|---|---|---|
| 6.1 | NAMR-only 嵌入的聚类 GO 富集分析 | `scripts/70_go_enrichment.py` | 每簇 Top GO term 超几何检验 |
| 6.2 | KEGG 通路成员身份 AUROC | `scripts/71_pathway_membership.py` | 各臂 × 多种子通路 AUROC |
| 6.3 | C3L 目标可恢复性探针 | `scripts/72_c3l_probe.py` | 线性探针 R² 分层分析 |
| 6.4 | 特征扫描探针 (PCA64 / Random) | `scripts/73_feature_scan_probe.py` | 控制输入特征信息含量 |
| 6.5 | C3L 损失置换检验 (B=1000) | `scripts/74_c3l_permutation_test.py` | 经验 p-value 计算 |
| 6.6 | 聚类诊断 | `scripts/75_cluster_diagnostics.py` | NMI / 机制控制 |
| 6.7 | KEGG 每种子 × 每臂成对差异 | `scripts/76_kegg_membership_table.py`, `scripts/130_kegg_per_seed.py`, `scripts/135_kegg_paired_diffs.py` | 配对差异检验 |
| 6.8 | Hold-out PPI 边预测评估 | `scripts/77_build_holdout_graph.py`, `scripts/78_eval_holdout.py` | 严格拆分下的真实 PPI 泛化 |
| 6.9 | 控制图构建 | `scripts/79_build_control_graphs.py` | expr_real / expr_shuffled / ppi_only 等诊断 |

### 2.8 Stage 7 — Result assembly, tables, figures | 结果汇总、表格与出图

**阶段目标**: 聚合所有实验结果，生成论文 LaTeX 表格与正式发表图表。

| 步骤 | 实验操作 | 脚本 | 输出 |
|---|---|---|---|
| 7.1 | 收集所有 checkpoint 的 per-epoch 损失历程 | `scripts/80_collect_losses.py` | `results/ablation_real/loss_summary.json` |
| 7.2 | 从原始制品重读所有数值并组装最终结果 | `scripts/81_assemble_results.py` | `results/final_results_summary.md` |
| 7.3 | 生成 LaTeX 格式主表与补充表 | `scripts/82_gen_final_tables.py` | `results/final_tables.tex` |
| 7.4 | 提取 Fig. 1(a) 网络子图布局（真实图） | `scripts/84_extract_fig1_layout.py` | `results/fig1_subgraph_layout.json` |
| 7.5 | 生成所有正式图表（PDF + PNG 双版本） | `scripts/85_make_figures.py` | `figures/` 目录图文件 |
| 7.6 | 图表规格审计（DPI、字体、边距） | `scripts/86_check_figure_specs.py` | 规格合规性报告 |
| 7.7 | 构建 Fig. 1 合成图 | `scripts/87_build_fig1_composite.py` | 最终合成 Fig. 1 |
| 7.8 | Hold-out 结果汇总 | `scripts/121_collect_holdout_results.py` | hold-out 评估汇总 |

### 2.9 Stage 8 — Benchmark & Audits | 基准测试与审计

**阶段目标**: 基准计算性能，审计手稿数据引用、参考文献、检查点完整性。

| 步骤 | 实验操作 | 脚本 | 内容 |
|---|---|---|---|
| 8.1 | 设备/BLAS 性能基准 (CUDA vs MPS) | `scripts/122_bench_device.py` | 前向/反向时间、内存占用 |
| 8.2 | 机制诊断（NAMR 降噪器、C3L 原型、CSP 几何捷径） | `scripts/93_*`, `scripts/94_*`, `scripts/125_*`, `scripts/126_*` | 机制细节诊断 JSON |
| 8.3 | 探针实验统计评估 | `scripts/124_eval_probe_regimes.py` | 多 regime 下 R² 比较 |
| 8.4 | Checkpoint 标准差评估 | `scripts/127_eval_ckpt_std.py` | 种子间稳定性量化 |
| 8.5 | 对照图性能报告 + 基线计算 | `scripts/123_*`, `scripts/128_*`, `scripts/129_*`, `scripts/131_*`, `scripts/137_*` | 控制条件完整报告 |
| 8.6 | 手稿数值与引用审计 | `scripts/88_*`, `scripts/89_*`, `scripts/90_audit_references.py` | 数值一致性 / 引用完整 |
| 8.7 | Checkpoint 冲突与完整性审计 | `scripts/132_audit_checkpoints.py` | 冲突检测 / 哈希注册 |
| 8.8 | 患者图修复（边一致性） | `scripts/133_fix_patient_graph.py` | 修正边方向一致性 |
| 8.9 | 辅助汇总（队列组成、跨种子几何、已知边、可复现性状态） | `scripts/100_*`, `scripts/101_*`, `scripts/102_*`, `scripts/103_*` | 多维汇总 JSON |

---

## 3. 实验结果 | Experimental Results

> 全部数值源自 `results/freezer/master_numbers_20261009.csv`（唯一数字底账）。

### 3.1 核心消融（seed 42，PPI known-edge AUC / few-shot 聚类 Acc）

| Arm | known-edge AUC | 聚类 Acc | pos_cos / neg_cos |
|---|---|---|---|
| `namr_only` | **0.8578** | 0.8025 ± 0.0160 | 0.7614 / 0.3094 |
| `csp_namr` | 0.7136 | **0.8799** ± 0.0126 | 0.7063 / 0.3557 |
| `full` (NAMR+CSP+C3L) | 0.5752 | 0.8519 ± 0.0193 | 0.5690 / 0.5152 |
| `csp_only` | 0.5237 | 0.8466 ± 0.0234 | 0.6505 / 0.6373 |
| `esm_only` (raw ESM-2) | 0.6221 | 0.7049 ± 0.0183 | 0.8935 / 0.8693 |
| `pca_input` | 0.7048 | 0.4648 ± 0.0417 | 0.1850 / −0.0019 |
| `random_init_hgt` | 0.6625 | 0.4882 ± 0.0220 | 0.9997 / 0.9994 |

### 3.2 五种子 headline（known-edge AUC，n=5，ddof=1）

| Arm | Mean ± SD | Seeds |
|---|---|---|
| `namr_only` | **0.8307 ± 0.0379** | 7, 21, 42, 99, 123 |
| `csp_namr` | 0.6352 ± 0.1107 | 5 种子 |
| `full` | 0.5930 ± 0.0699 | 5 种子 |
| `csp_only` | 0.5143 ± 0.0089 | 7, 42, 123（n=3） |
| `full_mps` (Apple MPS) | 0.6089 ± 0.0857 | 7, 42, 123（n=3） |
| `namr_only_mps` (Apple MPS) | 0.8348 ± 0.0352 | 5 种子 |

⚠ **注意**: `full − csp_namr` 的逐种子差值范围为 [−0.2166, +0.1477]，mean = −0.0423；seed 7 的 −0.2166 不得省略。CUDA 与 MPS 之间存在计算非确定性（如 seed 42 `full` 0.5752 vs 0.7077），已在 Limitations 中披露，严禁声称“100% 确定性复现”。

### 3.3 KEGG 通路归属（K=20 seed genes/pathway，30 draws，224 pathways）

| Arm | Mean ± SD（n=5） |
|---|---|
| `namr_only` | **0.8332 ± 0.0093** |
| `csp_namr` | 0.6178 ± 0.0941 |
| `full` | 0.5940 ± 0.0407 |
| `csp_only` | 0.5652 ± 0.0207（n=3） |

### 3.4 对照实验与关键机制诊断

- **3-hop 平滑基线（已知边）**: t=0 0.6074 → t=3 **0.9851** → t=10 0.9826；KEGG t=3 **0.9522**。平滑基线在 t=3 即大幅超越所有训练臂，作为正式强基线列入主表。
- **等预算对照（PPI AUC，n=3 seeds）**: `multi_cancer` 0.691 ± 0.115 优于 `single_brca` 0.562 ± 0.007；`eqvol`（4,998 边等体积子图）0.732 ± 0.065；`rewired`（105,000 边重连多重图）0.589 ± 0.044。Connectivity pattern 携带实质信息，重连图不优于等体积匹配子图。
- **C3L 原型退化**: 21 个癌种节点逐层 mean pairwise cosine 由 layer1 的 0.580 收敛至 layer6 的 1.0000，确认 prototype degeneracy 为主因（redundancy 机制不成立）。
- **NAMR 降噪器诊断（standardized）**: Wiener 线性去噪器 recon loss 0.2129 < identity 0.2648 < MLP（无图）0.2535 < graph_namr 1.0200 < constant_mean 1.0649，说明标准化后训练出的 NAMR 编码器逊于简单线性去噪器。
- **Hold-out PPI（13,622 边）**: `namr_only_heldout` random-neg AUC 0.8525 / degree 0.8086 / hub 0.7981；`csp_namr_heldout` 0.5952 / 0.5258 / 0.6401。

### 3.5 规模效应（650M 特征 / 13,881 基因，seed 42）

| 任务 | `namr_only` | `full` | `esm_baseline` |
|---|---|---|---|
| PPI AUC (5k eval) | 0.8962 | 0.7034 | 0.6706 |
| PPI AUC (full 13881 eval) | 0.8868 | 0.5871 | — |
| 聚类 Acc | 0.851 | 0.861 | 0.411 |

在 13,881 基因规模与 650M 特征下 NAMR 领先地位依然稳固，且图结构带来的通路归属能力增益随规模扩大而增强。

---

## 4. 图表文件 | Figure Inventory

所有图表均以 **PDF (矢量) + PNG (300 dpi)** 双版本发布，源文件位于 `figures/` 目录；图中所有数值从制品读取，无硬编码。

| 图号 | 文件基名 | 描述 | 数据来源制品 |
|---|---|---|---|
| Fig. 1 | `fig1_architecture` | Fig. 1 方法架构：(a) 异质图子网 (b) HGT 编码器 (c) 三任务联合预训练范式 | `results/fig1_subgraph_layout.json` |
| Fig. 2 | `fig2_convergence_v2` | 四消融臂 × 五种子的预训练损失收敛曲线 | 各 checkpoint `loss_history`, `loss_summary.json` |
| Fig. 3 | `fig3_ablation_v2` | 核心消融性能：(a) PPI AUC (b) 聚类 Acc (c) 多指标综合比较 | `results/ablation_real/eval_summary.txt` |
| Fig. S1 | `figS1_namr_reconstruction_fix` | NAMR 解码器修复前后性能对比（响应审稿意见） | `results/control/` 对照制品 |
| Fig. S2 | `figS2_patient_setfingerprint_tradeoff` | 患者级 P-NAMR set-fingerprint 权衡 | `results/patient_fewshot_acc.json` |
| Fig. S3 | `figS3_skipdecoder_collapse_diagnostic` | Skip-decoder vs Plain-decoder 塌陷诊断 | `results/figS3_auc_provenance.json` |
| Fig. S4 | `figS4_loss_curves_full_suite` | 全实验组损失曲线套件 | 各 checkpoint `loss_history` |
| Fig. S5 | `figS5_setfingerprint_and_phantomedge_mechanism` | Set-fingerprint 与 phantom-edge 机制 | `results/control/` 机制诊断 JSON |
| Graphical Abstract | `graphical_abstract_v1` | 图形摘要（16:9 = 1152×648 pt，三层带结构） | `scripts/83_make_graphical_abstract.py`（自包含设计常量 + 渲染宽度自检） |

数据溯源说明参见 `docs/FIGURE_REVIEW.md`；所有图表数据断言由 `tests/test_figure_data.py`（29 项检查）保障：**任何图数值改动必须同步更新制品并通过回归测试**。

### 4.1 出图硬性红线 | Figure Redlines

以下四条由 `scripts/86_check_figure_specs.py` 自动门禁（8 幅图逐图判定，任一不达标即 `EXIT=1`）：

| 红线 | 判据 | 当前实测 |
|---|---|---|
| 字体 | 最终 PDF 仅允许 Arial 家族子集；禁止 Helvetica / DejaVu / Type 3 | 8/8 图仅 `ArialMT`、`Arial-BoldMT`、`Arial-ItalicMT` |
| 字号 | 印刷后最小字号 ≥ 6.5 pt；matplotlib 设计字号取自 `TYPO_*` 常量 | 设计 7.0 pt → 印刷 6.7–7.0 pt |
| 分辨率 | PNG 元数据必须为 300 dpi（±0.5）；印刷有效 dpi ≥ 300 | 8/8 图 PNG 元数据 299.9994；Fig. 2/3 印刷 316 dpi，其余 300 dpi |
| 排版 | matplotlib 输出不得内嵌 `bbox_inches="tight"`；画布宽须等于 LaTeX `\includegraphics` 宽（scale = 1.0，supp 图为 `0.8\textwidth`） | 5/8 图 scale = 1.000；Fig. 1–3 因版式需要按 0.950/0.960 缩放 |

**数值零硬编码**：图中任何被绘制的数值必须经 `figure_data.py` 从 `results/` 制品读取，禁止在出图脚本中重复字面量（`tests/test_figure_data.py` + 门禁 S6 共同把关）。

✅ **已结清（v2.0.4）**：`graphical_abstract_v1.pdf` 原为 DejaVu Sans + `LastResortHE-Regular`（Type 3，含缺字回退字形 `⟂` U+27C2），违反上表字体红线。现已新建生成脚本 `scripts/83_make_graphical_abstract.py` 并重制：字体为纯 Arial 子集、零 Type 3、`⟂` 以 Arial 可用字符替代。详见 §6 v2.0.4。

---

## 5. 研究计划完成情况 | Research Progress Tracking

> 进度标记: ⬜ 未开始 | 🔄 进行中 | ✅ 已完成 | ⚠ 有阻塞 | ❌ 已取消

### 5.1 甘特式进度表 | Gantt-style Progress Table

| 阶段 | 任务 | 计划窗口 | 状态 | 完成度 |
|---|---|---|---|---|
| **Stage 1** | 公共数据下载 | 2026-08 W1 | ✅ 已完成 | 100% |
| **Stage 2** | 嵌入、图谱与目标构建 | 2026-08 W1–W2 | ✅ 已完成 | 100% |
| **Stage 3** | 预训练（4 臂 × 5 种子 × 2 平台） | 2026-08 W2–W4 | ✅ 已完成 | 100% |
| **Stage 4** | 核心评估（PPI AUC / Acc / 诊断） | 2026-08 W4 | ✅ 已完成 | 100% |
| **Stage 5** | 下游与外部验证（METABRIC / PAAD / GDSC2 / GTEx） | 2026-09 W1–W2 | ✅ 已完成 | 100% |
| **Stage 6** | 功能分析与机制探针 | 2026-09 W2–W3 | ✅ 已完成 | 100% |
| **Stage 7** | 汇总、表格与出图 | 2026-09 W3–W4 | ✅ 已完成 | 100% |
| **Stage 8** | 基准与审计 | 2026-09 W4 | ✅ 已完成 | 100% |
| **投稿准备** | 论文撰写与投稿（*Cell Reports Methods*） | 2026-09 W4 – | 🔄 进行中 | ~90% |

### 5.2 关键里程碑 | Key Milestones

| 里程碑 | 计划日期 | 实际完成日期 | 验收标准 | 状态 |
|---|---|---|---|---|
| M1: 预训练主数据收敛 | 2026-08-20 | 2026-08-21 | Full arm 5 种子均收敛 | ✅ 通过 |
| M2: 核心评估结果产出 | 2026-08-31 | 2026-09-01 | `eval_summary.txt` 覆盖所有 arm × 种子 | ✅ 通过 |
| M3: 外部验证完成 | 2026-09-15 | 2026-09-18 | METABRIC + PAAD 重跑确认 | ✅ 通过 |
| M4: 图表与回归测试通过 | 2026-09-25 | 2026-09-26 | 图表全部输出 + 43 断言通过 | ✅ 通过 |
| M5: GitHub 提交包就绪 | 2026-09-26 | 2026-10-16 | 本文档目录结构验证 | ✅ 通过 |

---

## 6. 变更记录 | Changelog

> 标记格式: ▲ 新增 ▼ 删除 ■ 修改 ⚠ 注意

### v2.0.4 — 2026-10-26（图形摘要重制：字体合规 + 九项版式缺陷修复 + 生成脚本入库）

| 区域 | 变更 | 操作人 | 说明 |
|---|---|---|---|
| `release/scripts/83_make_graphical_abstract.py` | ▲ 新增 | 项目组 | 新建 GA 生成脚本（此前仓库无任何 GA 生成代码）。自包含：Arial rcParams + `pdf.fonttype=42` 字体块（复刻 `85_make_figures.py` 模块级约定）、权威调色板常量、16:9 页面几何常量、数值 provenance 注释（`4,998` / `3.89M` / `5 seeds` / `500 epochs` / `96-row` / `10-command` / `5 verbs` 均标注制品来源）、以及**渲染宽度自检**（逐 span 度量，越界即 `SystemExit`） |
| `release/figures/graphical_abstract_v1.pdf`<br>`release/figures/graphical_abstract_v1.png` | ■ 修改 | 项目组 | 原图全文字以 **Type 3 `DejaVuSans` + `LastResortHE-Regular`** 嵌入（`⟂` U+27C2 在 Arial/DejaVu 中均缺字，被迫回退），违反 §4.1 字体红线。重制后：仅 `ArialMT` / `Arial-BoldMT` / `Arial-ItalicMT` 三个 CID TrueType 子集，**零 Type 3**；非 ASCII 字形仅保留 `· — ≠ ◦` 四种，均经 Arial 字形覆盖实测确认 |
| `release/Makefile` | ■ 修改 | 项目组 | `figures` 目标接入 `83_make_graphical_abstract.py`（置于 `85` 之后、`86` 门禁之前）；表头 `make figures` 说明补入 graphical abstract。GA 不纳入 `86`：`86` 依赖 `manuscript/__main.tex` 的 `\includegraphics` 清单，而扁平发布树无 `manuscript/`，故 GA 由脚本内建自检把关 |
| `release/project.md` | ■ 修改 | 项目组 | 版本升 v2.0.4；§4 图表清单补入 GA 生成脚本与画布规格；§4.1 原「已知未决项 ⚠」结清为「✅ 已结清」；新增本节 |

**GA 版式缺陷修复（D1–D9，与重制前 raster/字体取证一一对应）**

| 编号 | 缺陷（重制前实测证据） | 修复 | 复测结果 |
|---|---|---|---|
| D1 | METHOD 三列任务框内标签与说明叠印（最严重 22.03 pt 重叠） | 改为「主标签 1 行 + 灰色说明 1 行」两行式重排 | 37 span 自检 **0 overflow**；最紧 `task1-row1-desc` slack = +0.84 pt |
| D2 | 另两处任务框叠印（71.83 / 48.51 pt） | 同上 | 同上 |
| D3 | 中列主文本越出任务框右边线（x1 = 448.38 > 437.23，超 11.15 pt） | 两行式重排 + 逐列 `pick_size()` 二分缩字号 | 中列主文本字号 10.20 pt，slack ≥ +20.45 pt |
| D4 | 第 3 行 baseline（382.34）落到任务框底（381.57）之下 | 任务框底由 381.57 下移至 **394.00** | 末行墨迹止于 y = 382，框底线 y = 393–394 → **9 pt 净空** |
| D5 | Scope / Repro 框内表头与正文挤在一行（表头 489.12 vs 正文 487.65） | 表头独立成行（`CORNER_HEADER_BASE` 单独设定） | 表头与三行正文各自独立基线 |
| D6 | 三枚 Level 圆圈底色不一致（2 × `#1A5A3A` + 1 × `#1F8A4C`） | 统一为 `#1F8A4C` | 连通域检出 3 枚圆标，均为 12.72 × 12.72 pt、填充 `#1F8A4C` |
| D7 | 三层带标题贴带顶边 | 带标题下移（`BAND_TITLE_OFFSET = 15.5`） | 标题与带顶边净空 |
| D8 | METHOD 带标题与节点菱形右缘相撞（菱形右缘 284.62） | `METHOD_TITLE_X` 左移起点至 **293.60**；带框描边 `zorder = 3.0` 高于节点框 `2.0` | METHOD 标题 x0 = 293.60 > 284.62 |
| D9 | FINDINGS 带标题落入 guideline card1 框内 | 标题上提 + 三卡片整体下移（`CARD_SHIFT = 14.0`） | 标题墨迹 y 413–423，card1 顶边 y 430.5 → **6.8 pt 净空** |

**验证证据（v2.0.4）**

| 检查项 | 命令 | 结果 |
|---|---|---|
| 排版自检 | `python scripts/83_make_graphical_abstract.py` | `37 text spans checked, 0 overflow(s)` |
| 字体合规 | `pdffonts figures/graphical_abstract_v1.pdf` | 3 行：`Arial-BoldMT` / `Arial-ItalicMT` / `ArialMT`，全部 `CID TrueType`、`emb=yes`；**Type 3 = 0** |
| 缺字回退 | `pdftotext … \| 非 ASCII 字符统计` | 仅 `·` (U+00B7)、`—` (U+2014)、`≠` (U+2260)、`◦` (U+25E6)；无 `⟂` / `LastResort` 残留 |
| 页面尺寸 | `pdfinfo figures/graphical_abstract_v1.pdf` | `Page size: 1152 x 648 pts`（16:9） |
| PNG 分辨率 | PIL `Image.open(...).info["dpi"]` | `(4800, 2700)` px，元数据 **299.9994** dpi |
| 忠实复刻（形状清单） | `fitz.get_drawings()` 原件 vs 新件，`area ≥ 300` 逻辑形状 | 几何逐项一致（三层带 `[152.93, 78.31, 1027.87, 224.12]` 等）；fill / stroke 调色板完全一致；原件 23 个逻辑形状 → 新件 26 个（三层带 fill 与 stroke 拆为两个对象，+3） |
| 忠实复刻（文字清单） | 原件 span dump vs 新件 `pdftotext` | 69 个 span 文本逐条保留（`(-> ` 与 `<>` 为原件既有 ASCII 写法，未作改动） |
| 视觉边框 | 全页墨迹 bbox | x `[125.28, 1028.88]`、y `[77.28, 571.44]` pt；`x = 125.28` 为 TCGA 菱形左顶点，**属原件既有设计**（原件同坐标 `x[126.14, 223.24]`），非新增越界 |
| 流水线接入 | `make -n figures` | 顺序 `84 → 85 → 83 → 86`，命令展开正确 |

⚠ **注意（有意保留的差异）**：GA 中 `graphical_abstract_v1` 的数值（`4,998` 边 / `3.89M` 参数 / `5 seeds` / `500 epochs` 等）以**注释标注 provenance** 的方式写在 `83_make_graphical_abstract.py` 内，未走 `figure_data.py`。原因是扁平发布树不含 `results/` 制品目录，GA 又是对外唯一入口不能因缺制品而无法重生成；这与 §4.1「数值零硬编码」契约的偏差**已在此显式披露**，待 PI 裁定是否要求 GA 也改走制品读取。

### v2.0.3 — 2026-10-26（出图规格合规整改：Fig. 1 双版本对齐 + 字号泄漏清理 + 门禁修复）

| 区域 | 变更 | 操作人 | 说明 |
|---|---|---|---|
| `release/scripts/85_make_figures.py` | ■ 修改 | 项目组 | ▼ 删除 13 处写在 `set_title`/`set_xlabel`/`set_ylabel` 字符串字面量内的 `fontsize=N`——字面量会随图进入 PDF 文本层，污染图形文字并绕过 rcParams 字号契约；■ 2 处硬编码改为数据/常量来源（画布高 `2.95` → `FIG2_FIGSIZE`；skip-decoder AUC `0.601` → `figure_data.load_skip_decoder_auc()`）；■ 2 个名实不符的常量 `S3_SUPTITLE_FONTSIZE_A/B`（名为字号、实为标题文本）→ `S3_TITLE_A/B` |
| `release/scripts/figure_data.py` | ▲ 新增 | 项目组 | 新增 `load_skip_decoder_auc()` 与 `AUC_PROVENANCE_SKIP_PREFIX`，使 Fig. 2(b) 与 Fig. S3(b) 从同一制品 `results/figS3_auc_provenance.json` 读取同一训练运行的 AUC，避免两处各自复述数值 |
| `release/scripts/86_check_figure_specs.py` | ■ 修改 | 项目组 | ■ 修复 `designed_font_sizes()` 正则（原按 rcParams 字符串匹配，而实际字号定义为 `TYPO_*` 符号常量 → 触发 `AttributeError` 使门禁完全不可用）；▲ 新增 `FIGURE_SPECS` 单一清单（manuscript 提交名 → `85_make_figures.py` 中 builder 函数名），取代原先用旧图名的 `order` 列表；■ Fig. 1 位图的画布宽改按矢量 PDF 页盒设计宽（502 pt = 6.9722 in）度量，不再被 600-dpi 放大栅格的像素宽误判 `scale`；■ 位图 PASS 标签由无条件覆盖改为 `is_bitmap and not bad`，真实失败（如 dpi 越界）不再被掩盖 |
| `figures/fig1_architecture.png`<br>`manuscript/fig1_architecture.png` | ■ 修改 | 项目组 | 原 PNG（5977×2299 @ 600 dpi）经 md5 比对与 `archive/root_cleanup_20261016/.../fig1_architecture_BEFORE_8PT_FLOOR_RAISE_V101.png` 完全一致，即 **V101 之前的陈旧栅格**：内含烘焙图注（"Fig. 1 \| Graph Foundation Model (GFM) architecture…"）、术语与当前矢量 PDF 不一致、宽高比 2.5998 亦与 PDF 页盒（502×217 pt，2.3116）不符。改由权威矢量 `fig1_architecture.pdf` 以精确 300 dpi 重新栅格化 → 2092×905 px、元数据 299.9994 dpi、声明宽 6.9733 in，与 PDF 同版同比例 |

**验证证据（v2.0.3）**

| 检查项 | 命令 | 结果 |
|---|---|---|
| 门禁总判定 | `GFM_PROJECT_ROOT=<repo> python scripts/86_check_figure_specs.py` | **0 figure(s) failing，EXIT=0**（8/8 图 PASS；整改前为 `1 figure(s) failing`，EXIT=1） |
| 图形文字污染 | `for f in figures/*.pdf; do pdftotext "$f" - \| grep -c "fontsize="; done` | 8/8 图为 **0**（整改前 Fig. 2 等含泄漏） |
| 字体合规 | `pdffonts figures/*.pdf` | 8/8 图仅 `ArialMT` / `Arial-BoldMT` / `Arial-ItalicMT` 子集；DejaVu 与 Type 3 均为 0 |
| PNG 分辨率元数据 | PIL `Image.info["dpi"]` 逐图 | 8/8 图 **299.9994**，门禁容差 ±0.5 内 |
| Fig. 1 双版本一致性 | `pdfinfo` vs PIL | PNG 2092×905 @300 dpi（6.9733 in，比例 2.3116）与 PDF 页盒 502×217 pt 同比例；栅格无边缘裁切（墨迹 bbox 内缩 ≥ 27 px） |
| Fig. 1 旧件留痕 | `md5 manuscript/fig1_architecture.png` | 旧件 `1ca77c11…` 已备份至 `/tmp/gfm_backup_fig1/`，可在 `archive/root_cleanup_20261016/` 内溯源 |

⚠ **同批新增未决项（未经 PI 确认，暂不执行）**：`figures/graphical_abstract_v1.pdf` 字体为 DejaVu Sans + `LastResortHE-Regular`（Type 3，含缺字回退字形 `⟂` U+27C2），违反 §4.1 字体红线；且仓库内查无其生成脚本（`grep -rIl "graphical_abstract"` 仅命中 `project.md` / `README*` / 归档日志），需 PI 决定是否授权新建生成脚本并重制。

### v2.0.2 — 2026-10-17（双语 README 与发布仓库独立化）

| 区域 | 变更 | 操作人 | 说明 |
|---|---|---|---|
| `release/README_zh-CN.md` | ▲ 新增 | 项目组 | 与 `README.md` 逐节对齐的中文说明（仓库结构 / 脚本命名 / 运行方式 / 10 命令配方 / 数据可用性 / 图表 / 测试 / 语言规范 / 引用）；首尾均附英文版返回链接 |
| `release/README.md` | ■ 修改 | 项目组 | 顶部新增双语切换块；目录树补入 `README_zh-CN.md`；§Language 由“唯一 CJK 文件”改为列出 `project.md` 与 `README_zh-CN.md` 两个 |
| 父仓库 `README.md` | ■ 修改 | 项目组 | 第 17、156 行 `file:///Volumes/...` 本地绝对路径改为相对路径（发布文档禁止本地绝对路径，共 3 处） |
| 父仓库 `.gitignore` | ■ 修改 | 项目组 | 新增 `release/` 规则：release 已独立为单独 Git 仓库，父仓库不再追踪 |
| `release/` 仓库 | ▲ 独立化 | 项目组 | 独立 `git init -b main` + 首个 commit `1fa230d`；推送至 https://github.com/tjogzt/graphssl-multicancer-gfm-release （Public，默认分支 `main`） |

**验证证据（v2.0.2）**

| 检查项 | 结果 |
|---|---|
| release 远程文件数 | 119（118 + `README_zh-CN.md`） |
| release 工作区状态 | 干净；`main` 与 `origin/main` 同步 |
| release 推送前安全扫描 | 无密钥 / 无 token / 无本地绝对路径（唯一命中为文档中“已移除绝对路径”的说明文字） |
| 父仓库 README 本地绝对路径 | 0（原 3 处） |
| 父仓库未跟踪项 | 0 |

### v2.0.1 — 2026-10-16（提交前验证与根目录清理）

| 区域 | 变更 | 操作人 | 说明 |
|---|---|---|---|
| `release/scripts/85_make_figures.py` | ■ 修改 | 项目组 | mathtext rcParams 行内注释中文改为英文；CJK 扫描归零 |
| `release/README.md` | ■ 修改 | 项目组 | §10 recipe 步骤 (10) 补入 `test_new_scripts.py`，与“43 项断言”口径一致（原仅 2 条命令=38 项） |
| `release/PIPELINE.md` | ■ 修改 | 项目组 | Stage 7 `85_make_figures.py` 描述由 “all six figures” 改为 “every main + supplementary figure” |
| `release/docs/RENAME_MAP.v2.md` | ■ 修改 | 项目组 | 基线描述 `github_submit/scripts` → `scripts/` |
| `release/docs/VERIFICATION_20260927.log` | ■ 修改 | 项目组 | 顶部新增 Scope note；`Repo root` 本地绝对路径改为占位符（历史日志，正文数值保留） |
| 仓库根目录 | ▼ 归档/删除 | 项目组 | 被取代目录（`github_submit/`、`github_final_submit/`、`subject1_repo/`、`code/`、`drafts/`、`reviews/`、回滚备份）移入 `archive/root_cleanup_20261016/` 并 gitignore；清除全部 AppleDouble / `tmp-pdfcrop-*` / pytest-cache 残留 |
| `.gitignore` | ■ 修改 | 项目组 | 新增 `._*`、`tmp-pdfcrop-*`、`pytest-cache-files-*`、`archive/root_cleanup_*/` 规则 |

**验证证据（v2.0.1）**

| 检查项 | 结果 |
|---|---|
| `py_compile`（76 个 Python） | PASS 76/76 |
| 回归测试 | PASS 43/43（9 + 5 + 29） |
| 代码文件 CJK 字符 | 0 |
| 代码内绝对路径 | 0 |
| `@file` 头与文件名一致 | 0 处不符 |
| 动态 loader 目标存在性 | 4 处引用，0 缺失 |
| README/PIPELINE 脚本引用存在性 | 49 个引用，0 缺失 |
| `release/` 顶层条目 | 15（5 目录 + 10 文件），无 stray artifacts |

### v2.0.0 — 2026-10-16（发布目录标准化）

| 区域 | 变更 | 操作人 | 说明 |
|---|---|---|---|
| `release/project.md` | ▲ 新增 | 项目组 | 依据 Rule 13 重建权威文档，数值全部对齐 `master_numbers_20261009.csv` |
| `release/README.md` | ■ 修改 | 项目组 | 目录树/脚本计数校正为 81 文件（76 Py + 1 R + 4 sh）；新增 §Language |
| `release/docs/*` | ■ 修改 | 项目组 | 清除 `subject1_repo/`、本地绝对路径与 `manuscript.tex` 过时引用 |
| `release/scripts/120–137` | ■ 修改 | 项目组 | 修正 13 处 `@file` 头错位与 `Run:`/交叉引用（对齐 RENAME_MAP.v2） |
| `release/docs/PIPELINE.md`<br>`release/docs/Dockerfile.reference` | ▼ 删除 | 项目组 | 与顶层 `PIPELINE.md` / `Dockerfile` 完全重复且无引用 |

### v1.0.0 — 2026-09-26（初始创建）

| 区域 | 变更 | 操作人 | 说明 |
|---|---|---|---|
| `project.md` | ▲ 新增全文 | 项目组 | 权威文档首次创建，含课题目的/8 阶段路线/结果框架/图表清单/进度表 |

---

## 7. 版本管理规范 | Version Control Protocol

1. **Git 版本控制**: 每次文档更新必须提交，commit message 格式：
   ```
   docs(project): [变更摘要]

   ▲/▼/■ [具体变更项]
   Date: YYYY-MM-DD  Author: [名]
   ```
2. **修改标记格式**: 本文件内所有变更使用 `▲新增 ▼删除 ■修改 ⚠注意` 标记。
3. **历史版本追溯**: 通过 Git 历史保留所有旧版内容，禁止无记录覆盖。
4. **关联提交**: 文档变更必须关联对应代码/data 提交的 commit hash。

---

## 8. 质量保障措施 | Quality Assurance

1. **每周文档核查**: 项目负责人每周核查本 `project.md` 的完整性，重点检查 §3 结果与底账的一致性。
2. **关键节点会审**: M1–M5 共 5 个里程碑需团队会审，确认文档准确性后推进下一阶段。
3. **数据-代码-文档三角对应**:
   - 每个图表 → `docs/FIGURE_REVIEW.md` 数据溯源 → `tests/test_figure_data.py` 断言
   - 每个结果数值 → `results/**/*.json|txt` 原始制品 → 唯一数字底账 `master_numbers_20261009.csv`
   - 每个实验步骤 → `PIPELINE.md` 对应 Stage → `scripts/NN_*.py` 实际代码
4. **24 小时更新承诺**: 任何实验内容变更后 24 小时内必须同步更新本文档；重大调整填写 §6 变更记录。
5. **目录与关键词索引**: §0 目录索引、§1.0 关键词标签、章节锚点支持全文快速检索。

---

> **文档结束** — 如有疑问请联系项目负责人并参考 `README.md`、`PIPELINE.md`、`docs/` 子文档。
