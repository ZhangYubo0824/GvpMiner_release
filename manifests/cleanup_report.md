# GvpMiner 发布代码清理报告

## 当前范围

本报告记录从 `src3`、项目级 baseline runner 和论文结果分析脚本构建独立发布代码的决策。
源代码、原始数据和既有结果目录只读；发布包尚未配置 Git 远端。

## 第一阶段代码白名单

- 已检查源文件：773 个。
- 原路径保留：83 个。
- 迁移并按职责重命名：23 个。
- 排除：667 个。
- 发布运行文件：101 个 Python 文件和 5 个 shell 文件。

逐文件路径、处理决定和源/发布 SHA-256 见 `code_inventory.csv`。

## 已排除的整类内容

- `src3/tests`、Exp5 自带测试和 ESM 上游测试：仅用于开发验证，不属于最终运行代码。
- `src3/esm`：完整第三方仓库副本，含示例、notebook、测试和下载清单；发布包改用
  `fair-esm` 依赖，不重复分发第三方源码。
- `src3/exp3_control`：历史 T03 参考流程，不是论文采用的六种 baseline 入口。
- `src3/manuscript_tools`：Word 文稿修改器，不产生论文实验结果或论文图表。
- `__pycache__`、`.pyc`、编辑器配置、历史 walkthrough、设计/修改报告和一次性 QA 文件。

## 已迁入发布流程的项目级代码

- 六种 baseline 串行 runner、MotifAE 环境包装器和统一格式 adapters 已迁入
  `src/baselines`。
- motif 代表窗口、结构共同性、功能支持、文献区域审计和方法比较脚本已迁入
  `src/analysis`。
- 最新 v40 主图和补充图生成逻辑已迁入
  `src/publication_figures/generate_manuscript_figures.py`。

## 后续验证记录

路径解耦、函数级冗余审计、测试基线、最终结果重现和完整性检查将在后续阶段追加。
