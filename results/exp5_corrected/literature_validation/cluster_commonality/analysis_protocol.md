# Motif 簇内结构共性与外部未显著功能验证协议

## 分析日期

2026-06-08，时区 Asia/Shanghai。

## 一、结构分析

### 输入

- `result6/structures/esmfold/<family>/<sequence_id>.pdb`
- `result6/exp4_motif_structure_analysis/*/*/motif_structure_features.csv`
- `result6/exp3_domain_discovery/*/motif_representative_windows_*.csv`

### 方法

1. 使用 Exp4 保存的 `sequence_id`、`motif_start` 和 `motif_end`，将 motif 簇中的每个窗口映射到本地 ESMFold PDB。
2. 使用 PyDSSP 0.9.1 从主链 N、CA、C、O 坐标重新计算三态二级结构：
   - `H`：α 螺旋类结构；
   - `E`：β 链/β 折叠；
   - `C`：coil、loop、turn、bend 等非 H/E 结构的合并类别。
3. 对每个窗口计算 H/E/C 比例、最长连续片段、片段数、结构转换次数和压缩拓扑。
4. 对 motif 簇内全部可映射窗口进行汇总：
   - H/E/C 平均比例和标准差；
   - 以 H、E 或 C 为主的实例比例；
   - 相对位置二级结构共识；
   - 相对位置一致性得分；
   - 共识拓扑；
   - 共性结构类别。
5. 若同一 motif 簇中存在不同长度窗口，所有窗口均参与比例和主导结构统计；相对位置共识使用该簇的众数长度，并记录参与共识的实例比例。
6. 本分析只比较 motif 簇内部实例，不与随机窗口、邻近窗口或匹配对照窗口比较。

### 共性证据分级

- `high`：至少 10 个实例、映射覆盖率至少 0.8、位置一致性至少 0.8、平均 pLDDT 至少 70。
- `moderate`：至少 5 个实例、映射覆盖率至少 0.6、位置一致性至少 0.65、平均 pLDDT 至少 50。
- `low_or_uncertain`：不满足以上条件。

### 生物学意义解释

优先使用 Gvp 原始结构、突变和互作论文。GvpA、GvpC 和 GvpF 可与已有实验结构比较；其他附件蛋白主要属于预测结构共性加同源蛋白功能背景，不能视为 motif 的直接实验验证。

## 二、功能分析

### 输入

- 当前 `clusterprofiler/batch_results/*.csv`
- Exp3 motif catalog 和 representative-window 表
- 已核验的 Gvp 原始功能、突变、互作和工程论文

### 外部状态重算

所有状态直接从当前 clusterProfiler CSV 和日志重新计算，不沿用旧状态表。

- `p.adjust` 或 `qvalue <= 0.05`：`external_enrichment_supported`
- 有结果行但全部调整后 P 值大于 0.05：`external_enrichment_not_significant`
- 有 CSV、日志未提示基因集大小过滤失败、但 motif 没有结果行：`external_enrichment_no_terms`
- 日志出现基因集大小过滤失败：`external_filter_failed`
- 只有日志而没有 CSV：`external_enrichment_unavailable`
- CSV 和日志均无：`not_tested_external`

GvpK M2/M3 均存在 CSV，因此作为已有外部富集结果处理。GvpN M2 只有日志、无 CSV；GvpN M3 无 CSV 和日志，因此不进入“已有外部结果但未显著”的 motif 文献候选表。

### 文献检索

功能候选仅包括：

- `external_enrichment_not_significant`
- `external_enrichment_no_terms`

检索单位为 motif representative window，使用：

- family
- representative accession
- representative start/end
- representative 30-aa sequence

对所有候选进行了 Europe PMC 精确序列检索。相关 Gvp 功能文献用于判断该 motif 是否具有同源蛋白、邻近区域或家族级支持。没有把一般蛋白功能写成 motif 直接验证。

## 局限性

- ESMFold/PyDSSP 共性属于预测结构证据。
- 三态 DSSP 将 turn、bend 和其他非 H/E 结构合并为 C。
- 簇内一致性不能说明该结构相对于蛋白其他区域是否富集。
- 精确序列无文献命中不等于 motif 无功能。
- B 级证据表示同源蛋白、基因簇或邻近区域实验支持；C 级表示数据库或较弱结构推断。
- 只有针对该 accession 和该位置的突变、删除、结合或结构实验才能构成 motif 直接证据。
