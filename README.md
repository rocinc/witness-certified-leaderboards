# WITNESS

Certified pairwise leaderboards under sparse and judge-filled relevance labels.

[MIT License](LICENSE) (code) · CC BY 4.0 (data artifacts) · Python 3.12 · repository: `https://github.com/rocinc/witness-certified-leaderboards`

<details open>
<summary><b>中文</b>（默认展开；点击 English 切换）</summary>

## 简介

检索排行榜的每条成对结论均源自同类系统构建的相关性判断集。若抵消同排名文档，评测误差仅由“一方检出、另一方漏检”的差异文档决定。然而，池化判断集与大模型判官均会沿袭系统偏差：当全标注集池化偏差越过闭式阈值时，官方显著结论几乎尽数反向；随判官在词面与神经检索间的偏好变号，补全榜单在正确与失真间摆动。为此，本文提出的WITNESS按已知概率抽样候选文档进行人工评分，将判断集与判官作为减方差预测器；其估计对任意判断源无偏，在标签逐条到达时仍控制错误占比，并能预估预算认证对数，同预算下性能超越预测驱动与主动推断。在 DuRetrieval 上仅靠 843 条评分即可认证官方显著19对中的12对，定格词面与神经优劣，并为相近对比标出认证价格。

## 用一条命令复核主结果

```bash
pip install -r requirements.txt
bash scripts/reproduce.sh ./witness_data
```

这条命令不需要 GPU 或调用任何大模型，不需 API 密钥，依次完成五步：

1. 准备数据产物包（约 6 MB）并逐文件核对 md5。`./witness_data` 已存在时（例如随源码一并附带的解压副本）直接使用，不联网；不存在时从 GitHub Release 下载到该目录。
2. 运行单元测试，覆盖设计无偏性、超鞅性质、BH 程序与标注机制。
3. 用公开的人工标签重算 DuRetrieval 的可认证排行榜及其锚点校准版本，逐值比对 `results/figs/d3_results{,_anchor}.json`。
4. 重跑 TREC DL 2019 主表的第 0–1 次重复，逐行比对仓库中 40 次重复的完整结果表。每次重复的随机数由 `(seed, rep)` 唯一确定，所以部分重复与完整表中的对应行必须完全一致。
5. 从 `results/runs/` 重算全部图表数据，比对 `results/figs/fig_data.json`，再重画全部图。

全部通过时，最后一行输出：

```
REPRODUCED: D3 results, main-table repetitions and figure data match the released results
```

在线性代数库限制为单线程（`OMP_NUM_THREADS=1`）时，整条命令实测用时约 6 分钟，其中第 3 步的锚点校准多重插补约占 5 分钟。追加参数 `--full-d1` 会在 T2Ranking 上做同样的逐行比对；该测试床有 13 套系统和 48,640 个候选段落，原始运行日志中每次重复用时 6 到 14 分钟（服务器容器配额 16 核，多个评测进程同时运行）。

## 每个数字的出处

`docs/RESULTS.md` 为论文与报告中的每一个数字给出产生它的命令和输出文件。所有评估器结果表与日志都随仓库发布在 `results/runs/`，目录结构与运行时的 `outputs/` 相同，因此不运行任何程序也能逐项核对数字。

## 环境与依赖

全部结果产自 Ubuntu 22.04.5 LTS（x86_64，Linux 5.15）上的 Python 3.12.3，依赖的精确版本见 `requirements.txt`。只有编码与重排步骤需要 CUDA GPU（原始运行用一块 RTX 4080 SUPER，CUDA 13.0），从缓存的系统得分与判官等级往后的一切计算都在 CPU 上完成。

一键复核在同一系统上验证通过，所用工具为 GNU bash 5.1、GNU coreutils 8.32 的 `md5sum` 与 `tar`、curl 7.81。其他 Linux 发行版只要具备这几样工具即可运行；macOS 需先安装 GNU coreutils 以提供 `md5sum`；Windows 请在 WSL2 的 Ubuntu 中运行。`requirements.txt` 包含编码与重排所需的 torch，只做复核时可先装 CPU 版以节省下载：`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`，再执行 `pip install -r requirements.txt`。

随机种子全部写在配置文件里。评估器每次重复的随机数生成器由配置中的 `seed` 与重复编号 `rep` 共同初始化，DuRetrieval 第一批抽样的种子是 `configs/d3_duretrieval.yaml` 中的 `20261022`。每次运行会在日志开头打印完整配置和所运行代码的 md5。

## 数据

所有路径都相对于同一个根目录 `WITNESS_ROOT`。

- **数据产物包**（`witness_artifacts_v1.tar.gz`，GitHub Release 附件，CC BY 4.0）：测试床条目（只含 ID 与真值等级）、各系统得分、两个 DeepSeek 判官的等级、DuRetrieval 抽样框与各批次的入样概率，以及 DuRetrieval 人工评分。包内只有标识符、等级与数值，不含任何查询或段落原文。下载地址：`https://github.com/rocinc/witness-certified-leaderboards/releases/download/v1.0/witness_artifacts_v1.tar.gz`。
- **上游数据集**：T2Ranking、DuRetrieval（来自 DuReader-retrieval）、MS MARCO 与 TREC DL 2019，各自保留原始许可条款，需从各自的发布页获取。从原始数据完整重建时才需要它们。
- **人工评分**：DuRetrieval 两个批次共 843 条评分，另有 120 条锚点重标，存放于产物包的 `outputs/d3/labels/`。每个文件记录标注表行号与等级 0/1/2，行号经同目录的 `*_key.json` 对应到查询与文档。

## 从原始数据完整重建

按 `docs/RESULTS.md` 的"Inputs"表依次执行：构建测试床、对各系统编码打分（一块 RTX 4080 SUPER 上约 4.2 GPU 小时）、调用判官（DeepSeek 开放平台按挂牌价约 271 元）、运行评估器（`results/runs/logs/` 中 980 条重复记录合计约 26 个进程小时）。判官调用通过环境变量 `DS_BASE_URL` 与 `DS_API_KEY` 配置，代码不从仓库内的任何文件读取密钥。

## 目录结构

```
src/witness/      方法本体：估计量、抽样分配、认证程序、标注机制、数据加载
scripts/          实验入口：data/ 与 duretrieval/ 构建输入，eval/ 运行评估器，d3/ 处理人工标签，
                  diagnostics/ 与 theory/ 为机制诊断与理论核验，check/ 为复核用的比对工具，reproduce.sh 一键复核
analysis/         仅从保存的结果绘图（figkit/ 为统一的图表样式与排版检查）
configs/          测试床、评估与 DuRetrieval 配置
results/figs/     论文图表及其数据
results/runs/     评估器结果表与运行日志
docs/RESULTS.md   数字 → 命令 → 输出文件
tests/            保证性质的单元测试
```

## 许可

代码以 [MIT License](LICENSE) 发布；数据产物包以 CC BY 4.0 发布。两者都只覆盖本项目自己产出的代码与数据，各上游数据集与模型保留其原始条款，其中 MS MARCO 与 m3e-base 仅限非商业研究使用。

</details>

<details>
<summary><b>English</b> (click to expand)</summary>

## Overview

Every pairwise conclusion on a retrieval leaderboard is read off relevance judgments that systems like the two being compared helped to make. Sparse official judgments record only what earlier pools retrieved, and language-model judges that fill the holes make errors that move with which system retrieved a document, so "significant" conclusions can point the wrong way. WITNESS samples fresh human grades at known probabilities from every candidate document and uses the official judgments and the judges only as predictors that shrink the variance. Every pair of the leaderboard is tested at both ends of the scale for partial relevance and declared certified or open under control of the share of wrong conclusions, and before any label is bought the method forecasts how many pairs a budget will certify and what the open pairs would cost.

## One command

```bash
pip install -r requirements.txt
bash scripts/reproduce.sh ./witness_data
```

No GPU, no language-model calls and no API keys are needed. The command

1. prepares the data-artifact bundle (about 6 MB) and checks every file against its md5: an existing `./witness_data` is used as it is, with no network access (for example an unpacked copy shipped with the source), otherwise the bundle is downloaded from the GitHub release into that directory;
2. runs the unit tests of the guarantees (design unbiasedness, the supermartingale property, BH, the judging mechanisms);
3. recomputes the certified DuRetrieval leaderboard and its anchor-calibrated version from the released human labels and compares them value by value with `results/figs/d3_results{,_anchor}.json`;
4. reruns repetitions 0–1 of the main table on TREC DL 2019 and compares them row by row with the released 40-repetition table (every repetition is seeded by `(seed, rep)`, so a subset reproduces its rows of the full run exactly);
5. recomputes the figure data from `results/runs/`, compares it with `results/figs/fig_data.json`, and redraws every figure.

When every check passes, the last line is

```
REPRODUCED: D3 results, main-table repetitions and figure data match the released results
```

With the linear-algebra libraries limited to one thread (`OMP_NUM_THREADS=1`) the command took about six minutes, five of them in the anchor-calibrated multiple imputation of step 3. With `--full-d1` the same row-by-row check also runs on T2Ranking; that testbed has 13 systems and 48,640 candidate passages, and a repetition took 6 to 14 minutes in the original run logs (16-core container quota, several evaluator processes running at once).

## Where each number comes from

`docs/RESULTS.md` maps every number of the paper and the report to the command that produces it and its output file. Every evaluator table and log is released in `results/runs/`, laid out as the `outputs/` directory of a run, so each number can be checked against its file without running anything.

## Environment

Every result was produced with Python 3.12.3 on Ubuntu 22.04.5 LTS (x86_64, Linux 5.15), exact versions in `requirements.txt`. A CUDA GPU is needed only to encode and rerank (the original runs used one RTX 4080 SUPER with CUDA 13.0); everything downstream of the cached system scores and judge grades runs on the CPU.

The one-command check was verified on the same system with GNU bash 5.1, `md5sum` and `tar` from GNU coreutils 8.32, and curl 7.81. Any other Linux distribution with these tools runs it; on macOS install GNU coreutils first for `md5sum`; on Windows run it inside Ubuntu under WSL2. `requirements.txt` includes torch for encoding and reranking; for the check alone, install the CPU build first to save the download, `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.

Every random seed is in a config file: the evaluator seeds each repetition's generator from the config's `seed` and the repetition number `rep`, and the first DuRetrieval batch was drawn with seed `20261022` of `configs/d3_duretrieval.yaml`. Every run prints its full configuration and the md5 of the code it ran at the top of its log.

## Data

All paths are relative to one root directory, `WITNESS_ROOT`.

- **Data-artifact bundle** (`witness_artifacts_v1.tar.gz`, attached to the GitHub release, CC BY 4.0): testbed items (identifiers and true grades only), system scores, the grades of the two DeepSeek judges, the DuRetrieval frame with the inclusion probabilities of every batch, and the DuRetrieval human grades. It holds identifiers, grades and numbers only, no query or passage text. Download: `https://github.com/rocinc/witness-certified-leaderboards/releases/download/v1.0/witness_artifacts_v1.tar.gz`.
- **Upstream collections**: T2Ranking, DuRetrieval (from DuReader-retrieval), MS MARCO and TREC DL 2019 keep their own terms and are obtained from their own release pages; they are needed only to rebuild from raw data.
- **Human grades**: 843 grades in two DuRetrieval batches and 120 anchor re-labels, in `outputs/d3/labels/` of the bundle. Each file records the sheet row and the grade 0/1/2; the matching `*_key.json` maps rows to queries and documents.

## Rebuilding from raw data

Follow the "Inputs" table of `docs/RESULTS.md`: build the testbeds, encode and score the systems (about 4.2 GPU hours on one RTX 4080 SUPER), call the judges (about 271 yuan at DeepSeek list prices), and run the evaluator (the 980 repetition records in `results/runs/logs/` add up to about 26 process hours). Judge calls are configured by the environment variables `DS_BASE_URL` and `DS_API_KEY`; the code never reads a key from a file in the repository.

## Layout

```
src/witness/      the method: estimators, allocation, certification, judging mechanisms, loaders
scripts/          entry points: data/ and duretrieval/ build the inputs, eval/ runs the evaluator, d3/ handles the
                  human labels, diagnostics/ and theory/ hold mechanism and theory checks, check/ holds the
                  comparison tools of the one-command check, reproduce.sh is the one-command check
analysis/         figures drawn from saved results only (figkit/ holds the shared style and layout checks)
configs/          testbed, evaluation and DuRetrieval configs
results/figs/     the paper figures and their data
results/runs/     evaluator tables and run logs
docs/RESULTS.md   number -> command -> output file
tests/            unit tests of the guarantees
```

## License

Code under the [MIT License](LICENSE); the data-artifact bundle under CC BY 4.0. Both cover only the code and data this project produced. Upstream collections and models keep their own terms, and MS MARCO and m3e-base are restricted to non-commercial research.

</details>
