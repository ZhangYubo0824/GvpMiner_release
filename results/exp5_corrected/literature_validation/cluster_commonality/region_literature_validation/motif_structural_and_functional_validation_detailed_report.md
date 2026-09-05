# 聚类 Motif 结构共性与外部未显著功能的区域级文献验证说明

**项目目录：** `D:\研究内容\Task1`  
**分析日期：** 2026-06-08  
**分析对象：** Gvp 蛋白家族的聚类 motif  
**文档性质：** 方法、操作流程、结果解释与复现说明

## 1. 分析目标

本次工作将 motif 验证拆分为两个相互独立但可汇合的任务。

1. **结构任务：**利用每条蛋白已经生成的 ESMFold 结构，将 motif 簇内的全部窗口映射回各自蛋白结构，寻找簇内共同出现的 α 螺旋、β 折叠和 coil/loop 等局部结构模式；随后将这些共性结构映射到已发表的实验结构或局部实验区域，解释其可能的生物学意义。
2. **功能任务：**以 clusterProfiler 外部富集结果为主，只研究已有外部结果但未显著或没有 term 的 motif；通过完整蛋白同源对齐，将代表窗口投影到论文实际进行过突变、缺失、截短或片段互作实验的区域，寻找数据库富集未捕捉但已有科研实验支持的局部功能。

本分析不改变 Exp4 和 Exp5 的原有统计逻辑，也不覆盖任何原始结果。结构共性仍由 motif 簇的全部窗口计算，代表窗口仅用于文献检索、同源序列对齐、坐标投影和结果展示。

## 2. 关键原则

### 2.1 禁止使用不合理的 core 表示

分析和检索均不使用以下字段：

- `core_start`
- `core_end`
- `core_length`

motif 的检索表示采用 Exp3 生成的真实代表窗口：

- `representative_accession` 或 `representative_parent_id`
- `representative_start`
- `representative_end`
- `representative_length`
- `representative_sequence`

### 2.2 代表窗口不替代簇内全部窗口

代表窗口只是 motif 簇的一个可追溯检索入口。结构共性、实例数、位置一致性、平均 pLDDT 等统计仍然来自簇内所有可映射窗口。

### 2.3 外部工具无结果不等于不显著

不同状态必须区分：

- `external_enrichment_not_significant`：已测试，有结果行，但调整后 P 值均大于 0.05。
- `external_enrichment_no_terms`：该批次有有效 CSV，但该 motif 没有结果行。
- `external_filter_failed`：日志提示基因集大小过滤失败等问题。
- `external_enrichment_unavailable`：存在日志但没有可用 CSV。
- `not_tested_external`：CSV 和日志均不存在。

只有前两类进入本次“已有外部结果但未得到显著功能”的局部文献验证。其他状态不能写成“不显著”。

### 2.4 家族功能不能替代 motif 功能

“GvpA 是主要壳蛋白”“GvpK 参与装配网络”等家族级结论，不能直接证明某个 30 aa motif 有相同功能。功能支持必须满足：

1. 文献存在局部突变、缺失、截短、片段互作或结构界面实验；
2. 代表蛋白与实验参考蛋白具有可接受的同源对齐；
3. motif 投影区域与实验区域发生重叠；
4. 对具体点突变证据，实验位点的氨基酸必须在代表蛋白中保留。

## 3. 输入数据

### 3.1 Motif 与代表窗口

- `result6/exp3_domain_discovery/*/motif_catalog_*.csv`
- `result6/exp3_domain_discovery/*/motif_representative_windows_*.csv`
- 各家族 `windows_metadata.csv`
- 聚类标签和优化后的窗口 embedding

### 3.2 结构数据

- `result6/structures/esmfold/<family>/<sequence_id>.pdb`
- `result6/exp4_motif_structure_analysis/*/*/motif_structure_features.csv`

本地共使用 25,437 个 ESMFold 蛋白结构文件。最终有 682,352 个 motif 窗口完成二级结构提取，结构映射覆盖率为 100%。

### 3.3 外部功能富集

- `result6/exp5_external_functional_enrichment/clusterprofiler/batch_results/*.csv`
- 对应 `.log`
- `motif_sets_long.csv`
- `term2gene.csv`
- `term2name.csv`
- eggNOG、InterProScan 和 KofamScan 原始输出

外部富集状态全部根据当前 clusterProfiler CSV 和日志重新计算，不沿用旧状态表。

### 3.4 完整蛋白序列

- `result6/exp5_external_functional_enrichment/input/all_gvp_proteins.fasta`

完整序列用于将代表窗口所在蛋白与实验参考蛋白进行全局对齐。不能只用 30 aa 窗口直接进行同源定位。

### 3.5 外部实验与数据库证据

证据主要来自：

- PDB/RCSB：7R1C、8GBS、4QSG；
- UniProt 参考序列与人工整理的区域注释；
- GvpA、GvpC、GvpF 的结构论文；
- GvpJ、GvpM 的突变和片段互作论文；
- GvpL/GvpK 装配互作研究；
- GvpU 相分离和气囊聚集研究；
- Europe PMC、PubMed、PMC 全文及出版社页面。

全部来源记录在 `evidence_sources.csv`，每个局部实验区域记录在 `regional_evidence_catalog.csv`。

## 4. Exp3 代表窗口的生成方式

代表窗口不是根据所谓 motif core 截取，也不是人工选择。

具体步骤如下：

1. 对同一 motif 簇中所有真实窗口的 embedding 做 L2 归一化。
2. 计算归一化 embedding 的簇质心。
3. 再次归一化质心。
4. 计算每个窗口到质心的余弦距离：`1 - cosine_similarity`。
5. 选择距离质心最小的真实窗口作为代表窗口。
6. 若多个窗口距离相同，则按 `window_id` 字符串顺序稳定选择。

因此，代表窗口表示的是“最靠近簇中心的真实观测窗口”，而不是一个人为拼接的共识片段。输出同时保留：

- `distance_to_cluster_centroid`
- `n_cluster_windows`

这使后续检索可以追溯代表性强弱和簇规模。

## 5. 结构共性分析流程

### 5.1 将全部窗口映射到 ESMFold 结构

对 Exp4 中每个 motif 实例读取：

- `sequence_id`
- `motif_start`
- `motif_end`
- `motif_length`
- `motif_sequence`
- `motif_mean_pLDDT`

随后打开：

`result6/structures/esmfold/<family>/<sequence_id>.pdb`

按照原窗口坐标从完整蛋白结构中提取相同区间。该步骤使用簇内全部窗口，而不是只分析代表窗口。

### 5.2 重新计算二级结构

使用 PyDSSP 0.9.1 从 PDB 主链坐标计算三态二级结构：

- `H`：α 螺旋及其他 helix 类状态；
- `E`：β strand/β sheet；
- `C`：coil、loop、turn、bend 以及其他非 H/E 状态。

对每个窗口计算：

- H/E/C 比例；
- 最长连续 H/E/C 长度；
- H/E/C 片段数量；
- 二级结构转换次数；
- 压缩拓扑，例如 `C-H-C-E`；
- motif 区间平均 pLDDT。

### 5.3 簇内共性汇总

对每个 motif 簇计算：

- `n_structure_instances`
- H/E/C 平均比例与标准差；
- 以 H/E/C 为主的实例比例；
- 各相对位置的多数二级结构；
- `consensus_secondary_structure`
- `consensus_topology`
- `position_consistency_score`
- `minimum_position_consistency`
- `mean_motif_pLDDT`

若簇内窗口长度不同，所有窗口仍参与总体结构比例和主导结构统计；相对位置共识使用簇内最常见的窗口长度，并记录参与该共识的实例比例。

### 5.4 共性结构类型

分类规则如下：

| 类型 | 判定含义 |
|---|---|
| `alpha_helix_dominant` | 平均 helix 比例至少 0.5，且至少 60% 实例以 helix 为主 |
| `beta_strand_dominant` | 平均 sheet 比例至少 0.5，且至少 60% 实例以 sheet 为主 |
| `coil_loop_dominant` | 平均 coil 比例至少 0.65，且至少 60% 实例以 coil 为主 |
| `helix_coil_mixed` | helix 与 coil 平均比例均至少 0.25 |
| `beta_coil_mixed` | sheet 与 coil 平均比例均至少 0.25 |
| `alpha_beta_mixed` | helix 与 sheet 平均比例均至少 0.20 |
| `mixed_or_heterogeneous` | 不满足以上共性模式 |

### 5.5 结构共性强度

| 等级 | 条件 |
|---|---|
| `high` | 实例数 ≥10；映射覆盖率 ≥0.8；位置一致性 ≥0.8；平均 pLDDT ≥70 |
| `moderate` | 实例数 ≥5；覆盖率 ≥0.6；一致性 ≥0.65；平均 pLDDT ≥50 |
| `low_or_uncertain` | 未满足以上条件 |

只有 `high` 和 `moderate` 共性进入后续结构生物学意义验证。

## 6. 结构生物学意义的区域级验证

仅根据“高 α 螺旋”或“高 β 折叠”给出通用解释是不充分的。本次新增了区域级文献映射。

### 6.1 完整序列全局对齐

对代表窗口所属的完整蛋白与已发表实验参考蛋白进行 Needleman-Wunsch 全局对齐：

- match：+2
- mismatch：-1
- gap：-2

输出：

- 全局序列 identity；
- query coverage；
- query 每个残基到参考蛋白残基的映射；
- motif 区间 identity；
- motif 映射覆盖率。

### 6.2 坐标体系

Exp3 代表窗口使用 **0-based、右端不包含** 坐标：

`representative_start : representative_end`

论文、PDB 和 UniProt 区域通常使用 **1-based、两端包含** 坐标。

因此不能简单用一个固定偏移换算。程序通过序列比对逐残基建立映射，并从代表窗口覆盖的 query 残基集合计算参考蛋白的实际起止位置。

### 6.3 映射阈值

- 全局 alignment identity ≥0.25；
- 全局 query coverage ≥0.60；
- motif mapping fraction ≥0.80；
- motif 投影区域至少有 1 个残基与整理的实验区域重叠。

### 6.4 文献支持状态

- `homologous_region_supported`：motif 映射到同源蛋白已解析或已实验测试的局部区域。
- `no_region_specific_literature_mapping`：未找到可映射的实验结构或局部实验区域。
- `representative_full_sequence_missing`：无法取得代表窗口所属完整蛋白序列。

这里的“supported”表示同源区域支持，不表示该代表 accession 或精确 30 aa 窗口已经被直接实验验证。

## 7. 外部富集状态审计

共重新计算 18,854 个 motif 的 clusterProfiler 状态：

| 状态 | 数量 |
|---|---:|
| `external_enrichment_supported` | 11,662 |
| `external_enrichment_not_significant` | 1,900 |
| `external_enrichment_no_terms` | 684 |
| `external_filter_failed` | 1,797 |
| `external_enrichment_unavailable` | 1,449 |
| `not_tested_external` | 1,362 |

显著性优先使用 `p.adjust`，缺失时使用 `qvalue`：

- 调整后 P 值 ≤0.05：外部富集支持；
- 有 term 但全部 >0.05：外部富集不显著；
- 有有效 CSV 但 motif 无 term：已测试但无结果 term。

GvpK 的 M2/M3 均有 clusterProfiler CSV，因此作为“已有外部结果”的家族继续分析。GvpN M2 只有日志、M3 无 CSV 和日志，不能进入“已有结果但不显著”的功能候选集合。

## 8. 功能文献验证流程

### 8.1 候选集合

功能候选只包括：

- `external_enrichment_not_significant`
- `external_enrichment_no_terms`

共 2,584 个 motif。

### 8.2 为什么不以 30 aa 精确搜索为主

对全部候选进行过 Europe PMC 精确 30 aa 序列审计，命中数为 0。原因包括：

- 论文通常不在标题、摘要或正文中书写完整 30 aa 序列；
- 实验通常按突变位点、缺失长度、蛋白片段或结构域描述；
- 不同物种同源蛋白存在替换和插入缺失；
- 一个聚类 motif 不等于文献中命名的线性 motif。

因此精确序列检索只保留为负向审计，不作为主要证据发现方法。

### 8.3 区域级实验检索

主要检索以下局部证据：

- point mutation / alanine scanning；
- deletion / truncation；
- fragment interaction；
- split-GFP 或 pull-down 界面定位；
- shell stability / collapse pressure；
- gas vesicle formation phenotype；
- phase separation / clustering；
- display、fusion 或 engineering 的局部构建证据。

### 8.4 功能命中条件

1. 候选 motif 的完整蛋白与实验参考蛋白完成全局对齐；
2. 满足 identity、coverage 和 motif mapping 阈值；
3. motif 投影范围与实验操作区域重叠；
4. 若证据是一个具体残基或一组具体残基，代表蛋白必须在对应位置保留相同残基。

### 8.5 功能结果状态

- `homologous_region_supported`：映射到同源蛋白被局部实验操作过的区域。
- `no_localized_support`：没有发现符合条件的局部实验区域。
- `representative_full_sequence_missing`：缺失完整代表序列。

功能部分不设置 A/B/C/D 或其他证据等级。表中保留：

- `evidence_scope`
- `experiment_types`
- `source_ids`
- `limitations`

它们比单一等级更准确地表达证据边界。

## 9. 结构分析结果

### 9.1 总体结构共性

| 指标 | 数量 |
|---|---:|
| motif 结构窗口 | 682,352 |
| motif 簇 | 17,939 |
| 高共性 | 8,322 |
| 中等共性 | 6,460 |
| 低或不确定 | 3,157 |
| 进入结构文献验证的高/中共性簇 | 14,782 |

共性结构类别：

| 结构类别 | 数量 |
|---|---:|
| α 螺旋主导 | 9,416 |
| helix/coil 混合 | 3,601 |
| coil/loop 主导 | 2,245 |
| β strand 主导 | 1,439 |
| β/coil 混合 | 759 |
| α/β 混合 | 413 |
| 混合或高度异质 | 66 |

### 9.2 区域级文献映射

14,782 条高/中共性结构记录中：

| 状态 | 数量 |
|---|---:|
| 同源实验区域支持 | 3,368 |
| 无区域特异文献映射 | 11,411 |
| 代表完整序列缺失 | 3 |

各家族支持情况：

| 家族 | 高/中共性总数 | 区域支持 | 支持比例 |
|---|---:|---:|---:|
| GvpA | 704 | 422 | 59.9% |
| GvpC | 415 | 198 | 47.7% |
| GvpF | 715 | 703 | 98.3% |
| GvpJ | 3,401 | 1,855 | 54.5% |
| GvpL | 293 | 158 | 53.9% |
| GvpM | 46 | 32 | 69.6% |

其余家族暂时没有整理出可进行局部区域映射的实验结构或局部实验区域，因此仍保留簇内结构共性结果，但不能写成已有文献支持的具体生物学机制。

### 9.3 主要结构生物学结论

#### GvpA

GvpA motif 可映射到 7R1C 实验结构中的：

- N 端极性跨肋接触区；
- α1；
- β1-β2 hairpin；
- α2；
- C 端柔性未解析区域。

对应生物学意义：

- α1 跨越相邻 rib 间隙，参与 rib 间连接；
- β hairpin 构成气囊壁 rib 的结构核心并形成大量主链氢键；
- α1 与 β hairpin 的疏水残基构成面向气体腔的疏水内表面；
- 相邻 α1 之间形成气体扩散的狭缝；
- α2 构成外侧 GvpC 结合表面，关联壳体增强和 collapse pressure；
- C 端缺乏稳定密度时，应解释为柔性或异质区域，而不是刚性结构核心。

#### GvpC

GvpC 的高可信 motif 多为 α 螺旋主导，并映射到五个 33 aa 重复单元。结构论文、重复序列研究和突变实验共同支持：

- 重复单元沿 GvpA rib 外侧结合；
- 保守 Leu/Phe/Arg/Gln 等形成重复单元的结合面；
- GvpC 增加气囊壁有效厚度和抗塌陷能力；
- 重复单元具有一定冗余，不能把某一个重复 occurrence 单独解释为全部稳定功能。

#### GvpF

绝大多数 GvpF 高/中共性 motif 可映射到 4QSG 的两个实验 α+β 结构域。该结果支持：

- motif 位于稳定折叠结构，而非孤立无序肽段；
- 可能参与装配支架或蛋白互作表面。

但 4QSG 本身主要根据结构相似性和表面性质提出装配作用，不能将每个映射 motif 写成已验证的具体结合位点。

#### GvpJ

GvpJ 共性结构可映射到：

- 保守的 GvpA-like α-β-β-α 区；
- 独特、亲水的 C 端区域。

点突变和片段互作显示：

- α1 和 β 区域多处残基对气囊形成敏感；
- RAAIA/EYGL 邻近区域与 GvpL 互作及气囊稳定有关；
- C 端片段与 GvpG、GvpH 和 GvpL 等发生互作。

GvpJ 的二级结构来自预测/同源模型，而不是原子分辨实验结构，因此结论应写成“共性结构与实验敏感区域重叠”。

#### GvpM

GvpM 可分为：

- N 端 α-β-β-α 区；
- C 端 α3 区。

实验支持：

- N 端 1-25 aa 与 GvpL 互作并影响气囊形成；
- α1、GAV 和 RAAIA 区域存在突变敏感位点；
- C 端 60-84 aa 与 GvpF、GvpH 和 GvpJ 互作；
- 互作需要与气囊形成的必需性并不完全相同，较大 C 端删除仍可能保留部分形成能力。

#### GvpL

GvpL 的 133-281 片段保留与 GvpK 的互作。映射到该宽片段的 motif 可以解释为潜在装配界面候选，但不能缩小到具体残基。

## 10. 功能验证结果

### 10.1 总体结果

2,584 个外部未显著或无 term 的候选中：

| 状态 | 数量 |
|---|---:|
| 同源局部实验区域支持 | 465 |
| 未找到局部支持 | 2,119 |

支持结果全部来自 `external_enrichment_not_significant`。这表示这些 motif 所属集合在 clusterProfiler 中存在 term，但调整后 P 值未达到 0.05；局部文献证据提供了富集分析没有直接捕捉到的研究线索。

### 10.2 按家族统计

| 家族 | 功能候选 | 局部支持 | 支持比例 |
|---|---:|---:|---:|
| GvpJ | 490 | 315 | 64.3% |
| GvpL | 102 | 53 | 52.0% |
| GvpM | 49 | 22 | 44.9% |
| GvpA | 124 | 42 | 33.9% |
| GvpU | 64 | 18 | 28.1% |
| GvpC | 56 | 15 | 26.8% |
| GvpK | 1,223 | 0 | 0% |

其他家族没有发现满足局部实验映射条件的支持。

### 10.3 代表性结果

#### GvpA

例如 `GvpA_M2_motif_004`：

- 代表 accession：`UOT04069.1`
- 代表窗口：10-40
- 序列：`SSLADVIDTILDKGLVIDAYVRVSLVGIEL`
- n_unique_sequences：181
- 映射区域：GvpA N 端 22 aa 实验片段
- 候选功能：与 GvpF/GvpJ 的装配互作
- 证据：S06、S27

该结果表示 motif 与同源 GvpA 的局部互作片段重叠，不表示整个 30 aa 中每个残基都参与互作。

#### GvpJ

GvpJ 是局部支持数量最多的家族。候选 motif 可覆盖：

- J1-56 N 端实验片段；
- J47-114 C 端实验片段；
- α1-loop1 突变簇；
- RAAIA/EYGL 互作区域。

这类证据同时涉及气囊形成表型、气囊稳定性和 GvpL/其他装配蛋白互作，因此属于本次功能补充验证的重点结果。

#### GvpM

例如 `GvpM_M2_motif_004`：

- 代表 accession：`WP_167937484.1`
- 代表窗口：35-65
- n_unique_sequences：247
- motif alignment identity：约 0.85
- 映射到 RAAIA 区和 C 端互作片段；
- 候选功能涉及气囊形成、GvpL 互作以及 GvpF/GvpH/GvpJ 互作。

这是“外部富集不显著，但同源局部实验区域支持”中较强的代表性结果。

#### GvpL

GvpL 支持结果主要映射到 L133-281。该片段与 GvpK 互作，但区域较宽，因此更适合描述为“装配互作区域候选”，不能描述为精确 motif 界面。

#### GvpU

GvpU 证据来自五位点突变对相分离和气囊聚集的影响。分析额外要求代表蛋白保留相同实验残基后，仅保留 18 个 motif。

需要强调：

- 五个残基是联合突变；
- 不能把功能单独归因于某一个残基；
- 结果支持的是相分离/聚集相关区域，而不是单残基因果关系。

#### GvpC

GvpC motif 可映射到：

- C 端重复单元删除区；
- 五个重复中共同突变的保守结合面残基。

支持功能为壳体结合、机械稳定和 collapse-pressure 调控。多数支持的 GvpC motif 实例数较少，且重复单元具有冗余，应谨慎选择高实例数候选。

### 10.4 GvpK 的重要阴性结果

GvpK 有 1,223 个外部未显著或无 term 候选，但没有一个被标记为局部支持。

原因不是 GvpK 没有家族功能，而是目前整理到的实验只说明：

- 全长 GvpK 参与装配网络；
- GvpK 与 GvpL 的 C 端片段互作。

这些实验没有把互作位点定位到 GvpK 自身的局部区域。因此不能据此把某个 GvpK motif 判定为已有科研功能支持。该阴性结果体现了本次分析对 motif 局部证据的严格要求。

## 11. 如何解读最终结果

### 11.1 可以写入结果的表述

- “该 motif 簇在全部结构实例中呈现高一致性的 α 螺旋/β 折叠模式。”
- “代表窗口可映射到同源蛋白已解析的 α1、β hairpin 或 α2 区域。”
- “该同源区域在实验结构中参与 rib 连接、壳体核心或 GvpC 结合。”
- “该 motif 在外部富集结果中未达到显著性，但映射到同源蛋白经突变、缺失或片段互作验证的局部区域。”
- “该结果提示聚类可能捕捉到外部注释富集未覆盖的局部功能信号。”

### 11.2 不应写出的表述

- “该精确 30 aa motif 已被实验直接验证。”
- “所有落在宽片段中的残基都参与互作。”
- “外部无输出表示功能不显著。”
- “没有找到文献表示 motif 没有功能。”
- “GvpK 家族参与装配，因此所有 GvpK motif 都具有装配功能。”
- “联合突变证明某一个单独残基是功能必需位点。”

## 12. 输出文件说明

目录：

`result6/motif_validation/cluster_commonality_validation/region_literature_validation/`

| 文件 | 用途 |
|---|---|
| `structural_commonality_literature_validation.csv` | 全部 14,782 条高/中结构共性记录及文献映射状态 |
| `structural_commonality_region_supported.csv` | 3,368 条具有同源实验区域支持的结构结果 |
| `external_nonsignificant_motif_localized_function_validation.csv` | 全部 2,584 个功能候选的区域级文献审计 |
| `external_nonsignificant_motif_localized_function_supported.csv` | 465 个具有局部实验区域支持的功能候选 |
| `regional_evidence_catalog.csv` | 人工核验的实验区域、坐标、实验类型、功能和限制 |
| `evidence_sources.csv` | 33 个论文、PDB、UniProt 或本地数据库来源 |
| `search_protocol.md` | 区域映射阈值、纳入规则和局限性 |
| `results_summary.md` | 结果摘要 |

上级目录中的重要中间结果：

| 文件 | 用途 |
|---|---|
| `motif_window_secondary_structure.csv` | 682,352 个窗口的二级结构特征 |
| `motif_cluster_secondary_structure_commonality.csv` | 17,939 个 motif 簇的共性结构统计 |
| `external_enrichment_status_recomputed.csv` | 18,854 个 motif 的重算外部富集状态 |
| `external_enrichment_coverage_recomputed.csv` | 每个 family-method 的 CSV/日志覆盖审计 |
| `functional_motif_exact_sequence_search_audit.csv` | 精确 30 aa 文献搜索的负向审计 |

## 13. 关键字段说明

### 13.1 结构表

- `n_structure_instances`：参与结构共性统计的窗口数。
- `position_consistency_score`：各相对位置多数二级结构比例的平均值。
- `common_structure_class`：簇内共性二级结构类别。
- `mean_motif_pLDDT`：全部实例 motif 区域的平均 pLDDT。
- `literature_mapping_status`：是否映射到实验结构或局部实验区域。
- `mapped_reference_start/end`：代表窗口在实验参考蛋白中的投影坐标。
- `matched_region_labels`：命中的结构或实验区域。
- `literature_supported_biological_significance`：由该区域文献支持的生物学解释。

### 13.2 功能表

- `external_enrichment_status`：clusterProfiler 主导的外部富集状态。
- `min_adjusted_value`：该 motif 最小调整后 P 值。
- `top_nonsignificant_terms`：未显著 term 的前若干项。
- `localized_function_support_status`：是否存在局部实验区域支持。
- `candidate_function`：实验区域所支持的候选功能。
- `motif_alignment_identity`：代表窗口在映射参考区域中的序列一致性。
- `maximum_motif_overlap_fraction`：motif 投影区域与实验区域的最大覆盖比例。
- `minimum_reference_region_exact_match_fraction`：具体实验位点在代表蛋白中的保留情况。
- `evidence_scope`：证据是实验结构、同源局部区域还是宽片段。
- `limitations`：不能外推的范围。

## 14. 一致性验证

最终验证结果：

- 结构完整表：14,782 条；
- 结构支持子集：3,368 条；
- 功能候选完整表：2,584 条；
- 功能支持子集：465 条；
- 证据区域：63 条；
- 来源：33 条；
- motif 主键无意外重复；
- `source_ids` 均可在 `evidence_sources.csv` 中找到；
- 没有 `core_start/core_end/core_length` 字段；
- 功能结果没有 `evidence_grade` 字段；
- GvpU 来源 DOI/PMID 已校正并核验；
- 相关单元测试 14 项通过。

## 15. 局限性

1. 所有阳性结果均属于同源区域转移，不是对当前代表 accession 的直接实验。
2. 只要求实验区域与 motif 至少发生残基重叠；宽片段命中需要结合 overlap fraction 和 alignment identity 再排序。
3. GvpJ、GvpM 和 GvpL 的部分结构解释依赖同源模型或预测二级结构，不是原子分辨实验结构。
4. GvpC 重复单元高度相似，联合突变不能区分每个 repeat occurrence 的独立作用。
5. GvpU 五位点联合突变不能分解单个残基的因果贡献。
6. 未找到局部证据不表示 motif 没有功能，只表示当前纳入的文献中没有符合严格规则的局部实验。
7. clusterProfiler 的统计单位和 motif 局部区域实验回答的是不同层面的问题，二者不能相互替代。

## 16. 建议的后续优先级

1. 从 465 个功能支持 motif 中优先选择：
   - `n_unique_sequences` 较高；
   - motif alignment identity 较高；
   - motif overlap fraction 较高；
   - 实验区域较窄；
   - 同时具有高/中结构共性的 motif。
2. 对 GvpJ 和 GvpM 构建“突变位点-结构共性-外部富集状态”联合图。
3. 对 GvpA/GvpC/GvpF 制作 motif 到 PDB 区域的结构可视化。
4. 对 GvpK 设计局部截短或点突变实验，因为目前家族互作已知但 motif 局部证据缺失。
5. 对支持 motif 进行跨物种多序列比对，检查实验区域残基的保守程度。

## 17. 主要证据来源

- S01：Huber 等，2023，GvpA 气囊壳体 3.2 Å cryo-EM 结构，DOI `10.1016/j.cell.2023.01.041`。
- S02/S16：GvpF 晶体结构与 PDB 4QSG，DOI `10.1107/S1399004714021312`。
- S05：GvpM 片段与 GvpL/GvpF/GvpH/GvpJ 的 split-GFP 定位，DOI `10.3389/fmicb.2018.01897`。
- S06：GvpJ/GvpM 突变、气囊形成和互作实验，DOI `10.3389/fmicb.2021.794240`。
- S15：GvpU 相分离调控气囊聚集，DOI `10.1038/s41564-024-01648-3`。
- S21/S25：Anabaena flos-aquae 气囊结构与 PDB 8GBS。
- S22/S23：GvpC 33 aa 重复及重复删除对气囊性质的影响。
- S24：PDB 7R1C，GvpA 实验结构坐标。
- S26-S33：用于区域映射的 UniProt 参考蛋白序列。

完整来源及链接见 `evidence_sources.csv`。
