# 对比模型结构验证结果分析

数据来源：`result6/exp4_motif_structure_analysis/baseline`。该目录包含 16 个 Gvp 家族在 GLAM2、MEME、MotifAE 和 STREME 四个对比 motif 发现方法下的结构验证输出；每个 family-method 组合均包含 `analysis_summary.json`、`motif_cluster_summary.csv` 和 `structural_enrichment_results.csv`。为便于与正文主方法比较，本节同时参照 `result6/exp4_motif_structure_analysis/{family}/M2|M3` 的 GvpMiner 结构验证结果。

## 可直接写入正文的结果段落

为检验 GvpMiner 所发现候选 motif 的结构信号是否只是通用 motif 发现流程的共同结果，本研究进一步对四类对比方法产生的 motif 区间进行同一套 ESMFold 结构映射和 matched-control 检验。GLAM2、MEME、MotifAE 和 STREME 共请求 2,039,194 个 motif instance，其中 1,990,382 个成功进入结构分析，匹配得到 55,408,156 个对照窗口，并在 1,793 个 motif cluster 上完成 63,708 个“motif cluster × 结构 endpoint × 对照类型”的统计检验。总体上，1,768/1,793 个 baseline motif cluster（98.61%）至少在一个结构 endpoint 上达到 q <= 0.05，34,687/63,708 个 endpoint-level 检验（54.45%）达到 q <= 0.05。该结果说明，传统序列 motif 方法、概率/对齐式 motif 方法以及深度表示方法所产生的候选片段同样会携带可检测的结构偏倚；因此，结构验证应作为 motif 候选优先级排序的证据层，而不宜被解释为某一发现方法的单独充分证明。

不同对比方法呈现出不同的结构信号形态。GLAM2 的 cluster 数最少（160 个），但所有 cluster 均至少有一个显著结构 endpoint，endpoint-level 显著率为 67.60%，中位显著 endpoint 数为 23.5，且平均 motif pLDDT 较高（76.35）。这表明 GLAM2 倾向于输出较少但结构置信度和局部几何差异较集中的 motif。STREME 产生 529 个 cluster，所有 cluster 均出现显著 endpoint，endpoint-level 显著率为 60.41%，中位显著 endpoint 数为 22，显示出较好的覆盖度和结构一致性。MEME 产生 320 个 cluster，其中 296 个（92.50%）有显著结构 endpoint，endpoint-level 显著率为 53.32%，整体结构信号弱于 GLAM2 和 STREME。MotifAE 的 instance 规模最大（1,358,128 个已分析 instance）且 cluster 数最多（784 个），783 个 cluster（99.87%）出现显著 endpoint，但 endpoint-level 显著率为 48.28%，且跳过记录达到 47,943 个，主要来自缺失序列/结构和无效 motif interval。因此，MotifAE 结果更适合解释为高覆盖度候选空间，而不是单个 motif 边界更精确或结构效应更强。

从 endpoint 方向看，GLAM2、MEME 和 STREME 的显著结果集中在接触密度、半径、表面暴露、pLDDT 以及与蛋白质质心距离等特征上。三者的中位效应方向大体一致：motif 区域相较匹配对照通常具有略高的 contact density、较低的 surface residue fraction、较小的 radius of gyration，并且更靠近蛋白质质心；GLAM2、MEME 和 STREME 的 motif_mean_pLDDT 中位效应分别约为 +3.37、+1.68 和 +1.59。该模式提示这些序列 motif 更容易落在较紧凑、较中心或预测置信度较高的结构环境中。相比之下，MotifAE 的显著 endpoint 更集中于 relative_SASA、motif pLDDT、distance_to_protein_centroid 和 surface_residue_fraction，contact_density 显著检验数相对较少，且多个 endpoint 的中位效应接近 0。这提示 MotifAE 的高实例覆盖并不必然转化为更一致的局部 packing 信号，其候选片段可能混合了位置、表面暴露和模型激活边界等多类因素。

家族层面的结果显示，baseline 结构信号并非由单一家族驱动。GvpK、GvpJ、GvpO、GvpN、GvpA、GvpU、GvpC、GvpF、GvpT 和 GvpD 的 baseline cluster 显著率均为 100%；GvpL、GvpQ、GvpG、GvpH 和 GvpV 也保持在 93.02% 至 99.07% 之间。GvpM 的 cluster 显著率较低（76/88，86.36%），提示某些家族的 baseline motif 与结构 endpoint 的耦合程度较弱，或受候选边界、结构预测质量和样本规模影响更明显。按 analyzed motif instance 计，GvpK、GvpJ、GvpO 和 GvpN 是 baseline 结构分析中贡献最大的家族，说明这些大型家族在对比模型评估中具有较高统计权重，正文解释时应避免将其结果简单外推到所有 Gvp 家族。

与 GvpMiner 主方法相比，对比模型的结构验证更像是“窄候选空间中的结构富集参照”。GvpMiner M2 和 M3 分别产生 9,492 和 9,295 个结构验证 cluster，显著 cluster 比例分别为 94.65% 和 91.09%，endpoint-level 显著率分别为 68.57% 和 65.82%。相比之下，四个 baseline 合计只有 1,793 个 cluster，其中 GLAM2 和 MEME 的候选空间尤其小，而 MotifAE 虽有大量 instance，却集中到 784 个 feature-level cluster。由此可见，baseline 方法也能发现结构偏倚明显的 motif，但它们更多体现的是保守序列片段或方法特定特征的结构富集；GvpMiner 的优势不应表述为“所有结构指标均优于 baseline”，而应表述为在更大的候选空间内提供可追踪的结构筛选层，并与功能富集、同源区域文献证据共同形成候选 motif 的优先级排序框架。

因此，对比模型结构分析支持两点结论。第一，Gvp 蛋白中的候选 motif 区域普遍具有非随机结构环境，且这种现象可在多种 motif discovery 方法中复现，说明后续结构映射和 matched-control 检验是必要的。第二，不同方法输出的候选空间在规模、结构效应方向和质量控制风险上存在明显差异。GLAM2 和 STREME 更适合作为高置信、低数量的保守 motif 参照；MEME 提供传统 PWM 型序列 motif 参照；MotifAE 提供高覆盖度深度表示参照，但需要更严格处理无效 interval 和边界不确定性。结合这些结果，本文将 baseline 结构验证定位为方法学参照和解释边界，而将 GvpMiner 的主要贡献界定为大规模候选 motif 发现、结构/功能多证据筛选和实验验证优先级排序。

## 建议正文表述边界

- 可以写：baseline motif 也显示出非随机结构环境；结构验证是 motif 候选排序的共同必要环节。
- 可以写：GLAM2 和 STREME 的结构信号更集中，MotifAE 的候选空间更大但边界/QC 风险更高。
- 可以写：GvpMiner 的优势在于扩大候选空间并提供多证据优先级排序，而不是声称每个结构 endpoint 都压倒所有 baseline。
- 不建议写：baseline 结果证明某些 motif 具有真实生物功能。
- 不建议写：q 值极小等同于强生物效应；这些检验受样本量、cluster 大小、endpoint 数和对照设计影响。

## 可放入结果表的压缩数据

| 方法 | 已分析 motif instances | motif clusters | 显著 clusters | endpoint-level 显著率 | 中位显著 endpoint 数 | 平均 motif pLDDT | skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLAM2 | 258,410 | 160 | 160/160 (100.00%) | 67.60% | 23.5 | 76.35 | 202 |
| MEME | 122,354 | 320 | 296/320 (92.50%) | 53.32% | 19 | 69.87 | 78 |
| MotifAE | 1,358,128 | 784 | 783/784 (99.87%) | 48.28% | 17 | 72.49 | 47,943 |
| STREME | 251,490 | 529 | 529/529 (100.00%) | 60.41% | 22 | 71.43 | 589 |
| M2 (GvpMiner) | 370,069 | 9,492 | 8,984/9,492 (94.65%) | 68.57% | 24 | 72.18 | 289 |
| M3 (GvpMiner) | 335,042 | 9,295 | 8,467/9,295 (91.09%) | 65.82% | 24 | 72.12 | 271 |

## 补充材料建议

建议将完整 baseline 结构验证矩阵作为补充表，而正文只保留方法级摘要。补充表可包括：family、method、motif_cluster、n_instances、mean_motif_pLDDT、best_feature、best_control_type、best_q_value、best_effect_median_difference 和 significant_endpoints。这样可以避免正文被 1,793 个 baseline cluster 淹没，同时让审稿人能够追溯每个对比模型的结构信号来源。
