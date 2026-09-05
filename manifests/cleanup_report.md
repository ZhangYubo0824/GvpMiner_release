# GvpMiner 发布代码清理报告

## 交付范围与体积

本仓库是尚未推送的作者审阅包，只包含整理后的实验代码、`data3/ncbi` 的非 hybrid 原始
数据快照和论文采用的最终结果。源目录全程只读；中间结果没有被删除，只是不进入发布包。

- `src/`：89 个 Python、5 个 shell 和 1 个可编辑 SVG 源文件。
- `data/ncbi/`：53 个文件，101,102,171 字节（约 96.4 MiB）。
- `results/`：918 个文件，658,705,022 字节（约 628.2 MiB）。
- 数据与结果合计约 724.6 MiB；107 GB 是完整源目录校验扫描量，不是发布结果大小。
- Git LFS 当前管理 109 个大型 `.gp`/CSV；最大单个发布文件约 80.65 MiB。

## 代码白名单与内部清理

逐文件检查 775 个来源文件：原路径保留 73 个、迁移并按职责命名 22 个、排除 680 个。
完整决定、理由以及源/发布 SHA-256 见 `code_inventory.csv`。

保留的执行链覆盖数据预处理、Exp1–Exp5、六种外部 baseline、结构校验、校正后的功能富集、
precision-yield/Top-K、论文文献验证和最终 Figure 1–6/S1–S3 生成。所有新输出默认进入
`work/results/`，不会覆盖 `data/` 或 `results/` 快照。

已完成的内部清理包括：

- 删除无效导入、未使用变量/常量、重复导入、乱码日志和遗留绝对路径。
- 删除从 `src2` 一并复制来的测试、缓存、vendored ESM、T03 control、Word 修改器和历史文档。
- 删除与论文最终 v40 图重复且不兼容当前结果目录的旧通用绘图包；保留实际论文生成器。
- 将文献候选筛选器从一次性包装脚本解耦，直接调用核心区域映射函数。
- ProtBERT 默认使用与实验配置一致的 `Rostlab/prot_bert_bfd`，并优先复用
  `work/models/prot_bert_bfd`；模型、embedding 和 checkpoint 不纳入 Git。

## 数据与结果选择

NCBI 快照保留 23 个非 hybrid 家族的 `.gp`、FASTA 和 metadata，以及总表/来源表。
`GvpA_J_hybrid`、`GvpL_F_hybrid`、Python cache 和已迁入 `src/` 的提取脚本被排除；逐文件
哈希和排除理由见 `data_inventory.csv`、`data_exclusions.csv`。

最终结果按论文证据链分为八组：六方法 baseline 对比、Exp1、Exp2、Exp3、Exp4、校正后的
Exp5、precision-yield/Top-K 和论文图表。embedding、模型、PDB、逐窗口特征、matched controls、
批次、TERM2GENE、日志、进度状态与重复缓存均未上传。筛选明细见 `results_inventory.csv` 和
`results_exclusions.csv`；18 项论文图/表/核心结论映射见 `manuscript_result_map.csv`。

人工 curated literature source、experimental region、reference FASTA 和 search log 不属于作者
指定的 `data3/ncbi` 原始数据范围，未加入数据快照；对应脚本提供显式 CLI 参数，论文采用的
最终文献验证结果已保留。

## 验证记录

- 发布合同：13/13 通过（目录范围、无 hybrid、无测试/缓存、路径解耦、最终结果规则、文档等）。
- 静态检查：Ruff `F` 规则通过；Vulture 100% 置信度检查无未使用代码；全部 Python 文件可在
  内存中编译。
- CLI 冒烟：28/28 个实际命令入口的 `--help` 通过。
- 核心单元验证：从原测试中改指向发布 `src/` 的 8 个仍在范围内的模块，共 47 项通过。
  原测试全集未作为发布门槛，因为其中包含明确排除的 T03、旧绘图和已移除包契约。
- precision-yield/Top-K：按论文参数重跑；8 个核心 CSV/报告和 3 个绘图源数据 CSV 与冻结
  结果逐字节一致。manifest、PDF/PNG/SVG 的二进制会随运行路径、时间和 Matplotlib 版本变化。
- 论文图表：成功生成 9 个 SVG；5 个生成审计/源数据 CSV 与冻结文件逐字节一致。冻结和新生成
  SVG 均为有效 XML，且未嵌入 raster image。
- 源目录保护：重新计算 `data3/ncbi`、`result6`、`results7`、`analysis-output` 和
  `manuscript/v40_svg_figures` 的快照，全部与清理前 SHA-256 一致。
- 隐私与范围：未发现凭据模式、遗留绝对路径、`src2/src3/result6/results7` 运行时引用、测试或
  Python cache；Git remote 为空，未执行 push。
- 最终文件级校验值写入 `sha256sums.txt`，可在克隆或迁移后复核工作树内容。

## 当前环境限制

当前 Windows 主机没有可用的 Bash/WSL Linux 发行版，也没有本实验所需的外部工具和 GPU 模型，
因此未在此机重跑 MEME Suite、SLiMSuite、DiMotif、MotifAE、InterProScan、eggNOG-mapper、
KofamScan、clusterProfiler、ESMFold/AlphaFold 或完整 Exp1–Exp5。依赖和服务器运行方式已写入
`README.md`、`requirements.txt` 与 `environment.yml`；shell runner 将在目标 Linux 环境执行。
