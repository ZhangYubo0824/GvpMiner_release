# 候选级证据字段说明

主表每行是冻结 Stage C 20/15/M2 目录中的一个候选簇。蛋白、真实出现区间、结构终点、注释条目和文献事件各有独立明细表。布尔值是对应条件是否满足，空白表示缺失或不可计算；两者不能互换。没有人为加权总分。

|字段或字段组|定义与解释|
|---|---|
|motif_uid、family、method|唯一候选编号、蛋白家族和表示方法；连接使用完整候选编号，不能仅用数字簇标签。|
|catalogue_version、selection_mode|冻结目录版本及原发现阶段的选择方式；本次没有重新筛选522个候选。|
|cluster_label、rank_within_method、dominant_family|沿用原目录的簇标签、排序与家族记录；簇标签不用于跨模型对应。|
|min_start、max_end、mean_start、mean_end、position_std|原目录中支持窗口的位置摘要，不能代替每个成员的真实出现区间。|
|n_windows、n_unique_sequences、support_ratio|沿用原目录的支持窗口数、支持蛋白数与支持比例；本次确认支持蛋白ID去重数等于n_unique_sequences。|
|cross_species_count|仅为原字段留痕；全部522行均实际等于蛋白数，不作为物种多样性。新物种数使用tax_字段。|
|median_window_jaccard、median_protein_jaccard、median_occurrence_jaccard、median_recovery_fraction、stable_repeat_fraction、split_repeat_fraction、merge_repeat_fraction|原Stage C稳定性结果，作为冻结候选目录的既有字段保留。|
|seed_recovery_count、seed_total、pca_window_jaccard、null_empirical_p、null_fdr|原发现阶段的随机种子、PCA及空模型记录；本次未重新估计这些值。|
|n_occurrences|当前目录中实际出现区间数；一个蛋白可以贡献多个区间，因此不等于蛋白数。|
|representative_accession、representative_start_0based、representative_end_exclusive、representative_sequence、representative_selection|核对后的代表蛋白、0基半开区间[start,end)、该区间实际序列及代表选择来源；代表区间不概括整簇的全部位点。|
|structure_summarized、structure_testable、structure_supported|分别表示存在结构汇总、三个主要终点均有可用校正q值、可检验且至少一个主要终点q<0.05。|
|structure_strict|结构获支持且平均pLDDT≥70、组成匹配对照覆盖率≥0.9；不等于功能验证。|
|structure_state、structure_not_strict_reasons|分别保留支持状态与不满足严格支持的原因，包括未完成可检验条件、低pLDDT、低对照覆盖和已检验不显著。|
|mean_plddt、composition_control_coverage|原结构分析的平均pLDDT及组成匹配对照覆盖比例。|
|analyzed_occurrences、analyzed_occurrence_fraction、structure_occurrence_state|结构结果实际覆盖的出现区间数、占全部真实区间的比例以及complete/partial/missing状态。|
|contact_density_*、relative_SASA_*、long_range_contact_count_*|三个组成匹配主要终点；primary_q_value为主要检验校正值，effect_median_difference为候选减对照效应，ci95_low/high为原效应区间，n_proteins为原检验蛋白数。数值与符号原样保留。|
|annotation_testable、annotation_supported、annotation_status|原注释分析是否可检验、是否获富集支持及原始状态；支持需要可检验且存在p.adjust≤0.05的条目。不可检验与已检验不显著分开。|
|annotation_coverage、eligible_annotated_foreground_count|原注释公平性分析记录的注释覆盖率与合格已注释前景蛋白数；富集针对携带候选的蛋白集合。|
|significant_annotation_terms、minimum_annotation_p_adjust、representative_annotation_term|显著注释条目数、原分析最小p.adjust和显著条目中校正值最小的描述；完整条目保存在annotation_terms.csv。|
|verified_literature_events、verified_literature_members|通过现有人工审阅及当前序列/坐标复核的阳性文献事件数、对应成员数；只支持具体成员与位点，不支持整簇功能。|
|literature_status、literature_scope|文献支持的状态与范围；没有已审阅阳性映射不等于没有生物学功能，也不等于已完成穷尽检索。|

## 明细表

- occurrence_details.csv：以motif_uid、parent_id、occ_start、occ_end组成唯一键；原occurrence_id并非全局唯一。occ_length=end-start；support_window_ids保留原支持窗口；sequence为当前完整蛋白提取出的真实区间序列。
- structure_endpoints.csv：保留各终点效应、置信区间、原始p/q值、检验蛋白数和来源文件。候选编号位于motif_cluster；不能用终点条目数作为候选数。
- annotation_terms.csv：保留条目ID、Description、p.adjust、富集量、前景与背景规模以及相关蛋白ID；多个条目可以属于同一个候选。
- reviewed_literature_mappings.csv：保留A/B等级、原参考位点、目标位点、残基、文献、人工判读及当前区间复核。8条映射中含1条阴性表型；阳性事件计数不会包含该阴性记录。
- representative_cases.csv：保留全部有已审阅阳性位点映射的候选，以及各家族按编号确定的双重支持示例；不按本次MSA分数挑选例子。

## 合并主表中的新增前缀

|前缀|含义与分母|
|---|---|
|cross_|同家族、同蛋白实际残基集合的跨M2/M3区域对应。best为最佳区域交叠，hungarian_match_j050为主要阈值下的一一指派。M2与M3分母各自为522和464。|
|tax_|由organism字段得到的操作性物种/属标签、同家族背景覆盖、优势标签比例和标签解析比例。未知标签不作为新物种，但保留在全部成员蛋白分母。|
|msa_|家族MSA的不加权绝对保守性描述；mean/sd/median/min/max_conservation均针对每蛋白的有效出现得分中位数汇总；n_valid_proteins为有效蛋白数，覆盖率分母含不可评分出现。无背景、排名、百分位或显著性检验。单蛋白标准差为空，无有效蛋白分数为空。描述性结果已纳入v52。|

所有分析均属于发现后的计算证据。多项支持、富集显著、跨模型对应或恢复率都不构成因果功能验证。各模块methods/manifest及项目README记录具体参数、来源和复现命令。
