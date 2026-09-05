# GvpMiner 实验审阅包

这是供作者审阅的独立、尚未推送的 Git 仓库。它只包含三部分：整理后的完整实验代码、
`data3/ncbi` 的非 hybrid 原始数据快照，以及论文实际使用的最终结果。源目录中的中间结果
没有被删除；它们只是不进入本仓库。

发布包内 `data/` 和 `results/` 是只读快照。所有新运行默认写入 `work/results/`；可用环境
变量 `GVPMINER_OUTPUT_DIR` 指向服务器上的其他工作目录。Git LFS 管理较大的 `.gp` 和最终
CSV，克隆后先执行 `git lfs pull`。

当前 `results/` 共 918 个最终文件，658,705,022 字节（约 628.2 MiB），不是 107 GB；
`data/ncbi/` 共 53 个文件，101,102,171 字节（约 96.4 MiB）。约 107 GB 是交付前只读哈希
复核所扫描的完整源目录规模，其中包含明确不上传的中间结果。

流程依赖关系：

```text
NCBI -> preprocessing -> Exp1 -> Exp2
                    \-> Exp3 -> Exp4
                    \-> six baselines -> adapters -> Exp4
Exp3 + baselines + external annotations -> corrected Exp5
corrected Exp5 -> precision-yield/Top-K -> manuscript figures
Exp3/Exp4/Exp5 -> literature and validation summaries
```

## 环境安装

推荐 Python 3.12。论文图表记录的环境包括 Python 3.12.13、pandas 3.0.1/3.0.3、
matplotlib 3.10.9 和 seaborn 0.13.2；生成模型的配置记录 Transformers 4.46.3。

```bash
conda env create -f environment.yml
conda activate gvpminer
git lfs pull
```

`requirements.txt` 给出 Python 依赖。GPU 服务器应按 CUDA 驱动选择兼容的 PyTorch 构建，
再安装 `fair-esm`/ESMFold；模型权重写入 `work/models/`，不纳入 Git。Exp5 的富集阶段还需要
R 与 Bioconductor `clusterProfiler`。第三方来源和许可见 `LICENSES.md`。

## 数据准备

输入为 `data/ncbi/` 下的 23 个非 hybrid Gvp 家族目录；`GvpA_J_hybrid` 和
`GvpL_F_hybrid` 明确排除。仓库已保留原始 `.gp`、FASTA、metadata CSV、总表和来源表格。
如需从 `.gp` 重新提取 metadata，结果写到 `work/`，不会覆盖原始快照：

```bash
python src/data_collect/extract_gp_metadata.py --ncbi-dir data/ncbi --output-dir work/results/ncbi_metadata --overwrite
python src/data_preprocessing.py --cdhit-threshold 0.90
```

预处理执行质量过滤、精确去重/CD-HIT 去冗余和家族规模筛选，主输出为
`work/results/preprocessed/cleaned_gvp_dataset.csv` 及 `by_family/` FASTA。CD-HIT 不可用时可用
`--skip-cdhit` 明确回退。该阶段仅需 CPU；重复运行会重建目标输出。

## Exp1 家族分类

```bash
python src/exp1_classification.py --methods M2 M3
```

输入为预处理序列；M2 使用 ESM-2 `esm2_t33_650M_UR50D`，M3 使用 ProtBERT-BFD。
输出为交叉验证指标、预测、混淆矩阵、图和分类器工作文件。GPU 强烈推荐；已存在的 embedding
可复用，最终审阅快照仅保留指标和图表，不包含 embedding 或分类器 bundle。

## Exp2 生成与验证

```bash
python src/exp2_generation_validation.py --families GvpA GvpC GvpD GvpF GvpG GvpH GvpJ GvpK GvpL GvpM GvpN GvpO GvpQ GvpT GvpU GvpV --methods M2 M3
```

该阶段训练家族条件生成模型、生成候选序列，并用 Exp1 分类器验证；默认固定训练/生成参数可由
CLI 覆盖。GPU 推荐。模型、optimizer、训练文本与 embedding cache 属于中间工作文件；
`results/exp2_generation_validation/` 只保留最终生成序列、分类结果和汇总。除非指定
`--force-retrain`，已有完整模型目录可用于续跑。

## Exp3 motif 发现

```bash
python src/exp3_domain_discovery.py --methods M2 M3
python src/analysis/build_representative_windows.py --methods M2 M3
```

对每条序列使用 30 aa 窗口、5 aa 步长，分别计算 M2/M3 embedding，再进行 UMAP + HDBSCAN
聚类并生成 catalog、occurrence、代表窗口和聚类指标。embedding 计算需要 GPU；聚类和代表窗口
汇总可用 CPU。embedding、label 数组、窗口缓存和重复 JSON 未进入最终快照，CSV 与指标已保留。

## 六种 baseline 对比

六个外部 baseline 工具为 MEME、GLAM2、STREME、SLiMFinder、DiMotif 和 MotifAE；它们的输出
由 `src/baselines/adapters/` 转为统一 catalog/occurrence 结构。运行器需要 Linux/WSL Bash：

```bash
bash src/baselines/run_result_baseline_serial.sh --stage all --methods MEME GLAM2 STREME SLiMFinder DiMotif MotifAE
```

MEME、GLAM2、STREME 来自 MEME Suite；SLiMFinder 来自 SLiMSuite；DiMotif 与 MotifAE 需分别
提供 `DIMOTIF_CMD` 和 `MOTIFAE_CMD`。运行器在每个 family/method 目录检查完成标记并可续跑。
论文的最终六方法结构/功能图比较 M2、M3 与四个产生完整可比结果的 baseline（GLAM2、MEME、
MotifAE、STREME）；六个外部工具的执行和适配代码均保留，缺失/跳过情况由最终审计表记录。

## Exp4 结构分析

```bash
python src/predict_family_structures.py --backend esmfold
python src/exp4_motif_structure_analysis.py --from-exp3 --methods M2 M3 --num-random-controls 10 --seed 42
```

Exp4 使用全长 ESMFold/AlphaFold 结构，计算 motif 区域结构特征、长度匹配对照和 Wilcoxon/BH
统计；baseline 结构分析由 baseline 运行器调用同一流程。结构预测需要 GPU，已有结构默认复用；
分析进度可续跑。最终快照保留 enrichment、cluster summary、report 和 skipped audit，不包含 PDB、
逐窗口特征、matched controls 或 progress state。

## Exp5 校正分析

先准备统一蛋白索引与 motif set，再导入 InterProScan、eggNOG-mapper、KofamScan 的外部注释，
由 R/clusterProfiler 执行 family-method 富集。Kofam 校正仅保留达到阈值的星号记录：

```bash
python src/exp5_external_functional_enrichment/run.py prepare
python src/exp5_external_functional_enrichment/run.py parse-annotations --interproscan-tsv work/results/exp5_external_functional_enrichment/interproscan/all_gvp_proteins.fasta.tsv --eggnog-annotations work/results/exp5_external_functional_enrichment/eggnog/gvp_eggnog.emapper.annotations --kofamscan-tsv work/results/exp5_external_functional_enrichment/kofamscan/gvp_kofam.tsv
python src/exp5_external_functional_enrichment/run.py build-motif-sets
python src/analysis/analyze_functional_support.py
```

该阶段依赖 InterProScan、eggNOG-mapper、KofamScan、R 和 clusterProfiler；Python 汇总使用 CPU。
批次、TERM2GENE、注释缓存、日志和完成签名均为中间文件，最终快照只保留校正后的 merged
enrichment、support/fairness/Top-K 表及文献/区域校验结果。

论文的人工文献区域校验代码位于 `src/analysis/`。其 curated literature source、experimental
region、reference FASTA 和 search log 不属于本次指定的 `data3/ncbi` 原始数据范围，因此不在
数据快照中；重跑时须通过相应 CLI 参数提供。论文采用的最终文献校验表已包含在
`results/exp5_corrected/literature_validation/`。

## Precision-yield/Top-K 分析

```bash
python src/exp5_external_functional_enrichment/scripts/run_precision_yield_exploratory.py --motif-level-csv results/exp5_corrected/support_analysis/motif_level_fairness.csv --existing-random-summary-csv results/exp5_corrected/support_analysis/random_background_summary.csv --output-dir work/results/precision_yield
```

该 CPU 阶段固定 Top-K、bootstrap 和 family-method 内随机背景种子，输出 method yield、Top-K
敏感性、rank model、随机检验、源数据、图和 manifest。命令采用临时目录后原子替换已知输出，
拒绝覆盖含未知文件的目录。

## 论文图表生成

最终 Figure 1–6 与 Supplementary Figure S1–S3 均由冻结的 `results/` 源表生成：

```bash
python src/publication_figures/generate_manuscript_figures.py
```

输出写入 `work/results/publication_figures/`；加 `--qa-companions` 可另存 PDF 与 600-dpi TIFF。
SVG 保留可编辑文字并禁止嵌入 raster image。`results/publication_figures/` 中是论文采用的最终
SVG、源数据与 QA 记录，不会被生成命令覆盖。

## 结果与审计清单

- `manifests/code_inventory.csv`：从原代码到发布代码的保留、移动和排除决策。
- `manifests/data_inventory.csv` / `data_exclusions.csv`：NCBI 文件 SHA-256 与 hybrid 排除记录。
- `manifests/results_inventory.csv` / `results_exclusions.csv`：最终结果来源、用途、大小与 SHA-256。
- `manifests/manuscript_result_map.csv`：Figure、Supplementary Figure、Table 和 headline claim 到源文件及生成入口的映射。
- `manifests/cleanup_report.md`：代码清理、验证与已知环境限制。
- `manifests/sha256sums.txt`：除 `.git/`、`work/` 和该清单自身外，审阅包全部文件的 SHA-256。

当前包不含测试目录，这是作者明确选择的审阅范围；验证记录保留在 cleanup report。未配置
GitHub remote，也未执行 push。待作者审阅并提供私人仓库地址后再进行远端操作。
