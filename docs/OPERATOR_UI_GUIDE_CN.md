# HeymanDex VLA Data Engineering
# 数采人员操作与新电脑部署说明

本说明面向第一次接触项目的现场数采人员。请按顺序操作；遇到红色失败或不熟悉的技术原因，停止并联系算法/数据工程人员。不要为了让按钮变绿而改动原始数据。

## 1. 这个系统是做什么的？

它把已完成的真机数采变成经过检查、可发布的训练数据：真机数采 → RAW 原始数据 → 自动清洗 → 数据质量检查 → LeRobot 数据集 → 私有 Hugging Face（HF）数据集仓库。界面不负责 π0.5 训练、机器人推理或机器人控制，也不会代替数采程序录制数据。

## 2. 数据处理流程图

```text
数采完成
   ↓
选择数采批次（Run）
   ↓
物理数据检查 D2
   ↓
数据质量检查 D3
   ↓
生成 LeRobot
   ↓
本地验证
   ↓
HF 上传检查（Dry Run，只读）
   ↓
人工核对并确认
   ↓
HF 正式上传
   ↓
固定 Commit SHA 重新下载并远端验证 → 完成
```

前一步失败时不要跳过。黄色警告需要阅读 Episode 原因；绿色表示该检查完成，不等于所有下游步骤已完成。

## 3. 新电脑最低要求

- 本项目 `pyproject.toml` 要求 **Python >=3.12,<3.13**。当前已验证机器为 Ubuntu 24.04、Python 3.12.3、uv 0.11.26；这些是环境实测值，不表示所有机器必须使用完全相同的补丁版本。
- 需要 Git、uv、能打开本机网页的浏览器。准备 HF 发布时还需要网络、HF 账号和 `hf` CLI。
- 项目本身仅声明 NumPy >=2,<3、Pillow >=10,<13；开发依赖包括 pytest、ruff。项目依赖安装**不会自动准备** LeRobot/HF 专用 Python 环境。
- 需要可读写的数据盘和足够空间容纳 RAW、工作结果、LeRobot 视频以及 HF 合并/重新下载的临时副本。具体容量取决于批次大小，不给一个虚构的固定数字。请由管理员提前核对可用空间。
- 导出和发布还需要数据工程人员提供**已审计、现成**的 LeRobot Python 环境，满足代码中的 `CODEBASE_VERSION == "v2.1"`，并可导入 `huggingface_hub`。当前机器所用解释器见第 8 节；路径不能直接照搬到新电脑。当前解释器的发行包元数据为 `lerobot 0.1.0`、`huggingface_hub 0.36.2`，与数据格式标识 `v2.1` 是不同概念。

## 4. 推荐目录结构

```text
/home/<你的用户名>/heymandex-vla-data-engineering/  项目代码和项目 .venv
/data/vla_runs/                               各次数采批次与处理产物
/data/uv-cache/                               uv 下载/构建缓存
```

`/data/vla_runs/<批次名>/raw` 是原始输入；`work` 是系统生成的整理结果。不要把项目代码、缓存和数采 RAW 混成一个目录，也不要用个人桌面的历史验收路径作为新批次默认路径。

## 5. 克隆项目

先确认新电脑拥有该 GitHub 仓库的读取权限。仓库地址取自本项目当前 `git remote -v`：

```bash
cd "$HOME"
git clone https://github.com/YSJHYX/heymandex-vla-data-engineering.git
cd heymandex-vla-data-engineering
git status --short
```

`$HOME` 会自动使用当前账号的主目录。若 `git clone` 报权限错误，联系仓库管理员开通权限，不要使用别人的密码或 token。

## 6. Python / uv 环境配置

由管理员先安装 Python 3.12、Git 和 uv。uv 官方安装页为 <https://docs.astral.sh/uv/getting-started/installation/>；在可联网的新电脑上，可先检查官方安装脚本，再执行其说明的安装命令，例如：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
python3.12 --version
```

在**本项目**根目录，首次安装时可按锁文件创建 `.venv` 并安装项目及开发依赖：

```bash
cd "$HOME/heymandex-vla-data-engineering"
export UV_CACHE_DIR=/data/uv-cache
uv sync --frozen
.venv/bin/python --version
.venv/bin/python -c "import vla_data; print(vla_data.__file__)"
```

这里的 `uv sync --frozen` 只针对新电脑上的 **Data Engineering 仓库**；不要在 OpenPI 中运行，不要用它替代已审计 LeRobot 环境的准备。日常操作使用已有 `.venv`，不需要反复安装。若同步失败或锁文件提示不兼容，联系数据工程人员，不要自行升级/降级依赖或改锁文件。

## 7. `/data` 目录配置

先用 `df -h /data` 确认 `/data` 已挂载且空间充足，再建立两个目录：

```bash
mkdir -p /data/vla_runs
mkdir -p /data/uv-cache
```

若显示 `Permission denied`，表示当前用户对 `/data` 没有写权限，请管理员建立目录并把这两个**具体目录**的权限交给运行 UI 的账号；例如管理员在确认目标盘后执行 `sudo mkdir -p /data/vla_runs /data/uv-cache`，再按本机权限政策授权。不要对整个 `/data` 做递归改权限。`/data/vla_runs` 放数采批次和处理结果；`/data/uv-cache` 只是 uv 缓存，不放 RAW。

## 8. UI 配置文件 `config/ui.toml`

在新电脑上由数据工程人员核对并调整**路径配置**，不要在里面放 token。当前仓库示例值如下：

```toml
allowed_data_roots = ["/data/vla_runs"]
allowed_raw_roots = ["/data/vla_runs"]
state_dir = "/data/vla_runs/.vla_data_ui"
default_hf_repo = "PPPPPilot/VLADexData"
uv_cache_dir = "/data/uv-cache"
lerobot_python = "/实际存在的/已审计环境/bin/python"
hf_python = "/实际存在的/已审计环境/bin/python"
host = "127.0.0.1"
port = 8765
```

`allowed_data_roots` 限定 UI 可看到的批次根目录；`allowed_raw_roots` 限定 RAW 来源；`state_dir` 保存本机任务数据库/日志，不写到 RAW；`default_hf_repo` 是默认私有 Dataset Repo ID，不是登录凭据；`uv_cache_dir` 是缓存；`lerobot_python` 是可通过官方 LeRobot API 回读 v2.1 数据集的解释器；`hf_python` 是可调用 `huggingface_hub` 的解释器；`host` 必须保持本机 loopback，`port` 默认为 8765。

现有文件另外包含历史 `batch4_acceptance` 的只读 `run_overrides`：它专门指向旧机器桌面上的共享 RAW，并仅允许 003–006。**新电脑不要照搬这些历史路径**，应由数据工程人员按新机器实际情况处理；普通新批次不需要 override。配置路径可由 `--config` 选择；不要把 HF token 写入此文件。

## 9. Hugging Face 账号配置

### 9.1 注册账号

访问 <https://huggingface.co/>，注册并验证自己的账号。若团队统一由某组织或专用账号管理数据，先询问管理员应使用哪个身份。

### 9.2 获取 Access Token

登录 HF 网站，进入头像菜单的 `Settings` → `Access Tokens`。为目标**私有 dataset repo** 创建最小权限 token：优先使用限定该仓库的 fine-grained token，授予读取私有仓库和写入该仓库所需权限；只读 token 无法上传。不要授予与本任务无关的组织/仓库权限。官方权限说明：<https://huggingface.co/docs/hub/security-tokens>。

### 9.3 在终端登录

`hf` CLI 由管理员按官方安装说明准备：<https://huggingface.co/docs/huggingface_hub/main/guides/cli>。若新电脑没有该命令，可先核对 HF 官方 CLI 安装脚本，再按其说明执行：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf --version
```

确认 `hf --version` 能运行后，在**同一台电脑、同一运行 UI 的系统账号**下执行：

```bash
hf auth login
```

按终端提示粘贴 token。粘贴时终端可能不显示字符，这是正常的；不要把 token 当命令参数打印出来。UI 页面没有 token 输入框。

### 9.4 检查登录

```bash
hf auth whoami
```

应看到实际登录用户或组织，例如 `PPPPPilot`；不是这个身份就先联系管理员。CLI 登录只是基本检查，还须在 UI 的 HF 页确认目标仓库可读、为 Private、具有发布权限。配置的 `hf_python` 环境也必须能使用该账号的 HF 凭据。

### 9.5 Token 安全

不要把 token 发给别人、写进 README、写进 `config/ui.toml`、写进代码、贴到聊天、截图或任务日志。泄露时应在 HF 设置中撤销 token 并通知管理员。

## 10. 创建 HF Dataset Repo

先由具备权限的人员在 HF 网页创建 **Dataset** 类型仓库，Visibility 选择 **Private**，例如 `PPPPPilot/VLADexData`；不是 Model Repo 或 Space。仓库可以为空，但必须由当前账号可读且可写。UI 不会向 Public Repo 正式发布。如果发现已是 Public，不要继续，也不要随意迁移或删除；联系仓库管理员。

## 11. 如何放置数采数据

一次完整数采对应一个批次目录。真实 RAW 结构是 `.npz` 文件加同名 `_media` 图像目录，不是把每个 Episode 当顶层 Run：

```text
/data/vla_runs/20260917_ball/
└── raw/
    ├── episode_000000.npz
    ├── episode_000000_media/
    │   └── ...头部/腕部相机图像...
    ├── episode_000001.npz
    └── episode_000001_media/
```

在 UI 下拉框选择的是 `20260917_ball`，**不是** `raw`、`episode_000000` 或单个 `.npz/.json/.mp4` 文件。标准批次路径由系统自动推导：

```text
RAW_ROOT      /data/vla_runs/<run_name>/raw
WORK_ROOT     /data/vla_runs/<run_name>/work
CURATED_ROOT  /data/vla_runs/<run_name>/work/curated
QUALITY_ROOT  /data/vla_runs/<run_name>/work/quality
EXPORT_ROOT   /data/vla_runs/<run_name>/work/lerobot
```

确认数采程序已结束并完整交付文件后再处理；不要手动改名、补造或移动 RAW 内的文件。历史 `batch4_acceptance` 是特殊只读映射，它可能显示外部 RAW 路径；以 UI “RAW 输入”和配置为准。

## 12. 如何启动 UI

在项目根目录，用已有项目 `.venv` 启动：

```bash
cd "$HOME/heymandex-vla-data-engineering"
export UV_CACHE_DIR=/data/uv-cache
.venv/bin/python -m vla_data.ui
```

浏览器打开 <http://127.0.0.1:8765>。这是本机地址，其他电脑不能直接访问。要停止服务，在启动终端按 `Ctrl-C`；不要在处理中强制断电。若 `config/ui.toml` 不在当前工作目录，请先 `cd` 到项目根目录。页面左侧“使用帮助”可离线打开本说明。

## 13. Stage 1：选择数采批次

1. 找到“选择数采批次（Run）”下拉框，选本次完整批次名；例如 `20260917_ball`。
2. 核对“RAW 输入”路径应指向该批次的 `raw/`（历史只读批次例外）。不需要手填工作目录。
3. 看“原始数采 Episode”数量是否与交付数量大致一致；为 0 时停止。
4. 看 Task 预览，即每条轨迹的原始语言指令。发现空 Task 或内容明显不符时停止并报告。
5. 看头部、腕部相机 Episode 计数。缺一路时不要忽略。

## 14. Stage 2：数据清洗

点击“开始数据清洗”。后台依次运行 D2 物理/因果数据检查和 D3 质量检查，不修改 RAW。界面会显示处理状态和运行日志。绿色“通过”表示检查完成；黄色“有警告”表示部分数据可继续但必须查看原因；红色“失败/拒绝”表示该 Episode 不能直接用于后续训练。已有结果可复用，“重新运行”会影响下游产物，必须二次确认并由数据工程人员判断是否需要。

## 15. Episode 检查结果

表格逐条显示物理检查、质量检查、有效帧数和结果。点击“查看原因”会看到中文摘要，例如“缺少腕部相机数据”“没有可用于训练的有效数据区间”“状态和动作的时间顺序异常”。需要排错时再展开“高级信息”，查看技术代码和原始详情。未知技术代码会显示“未知数据问题”，不能据此自行猜测原因。不要编辑 RAW、补帧、用命令值冒充实测状态；把 Episode 编号和原始详情交给工程人员。

## 16. Stage 3：LeRobot 导出

点击“生成 LeRobot 数据集”，只会导出连续的有效区间，不会把被排除的中间段拼接起来。高级选项通常保持默认：验证集比例 `0.1`（约 10% 来源进入验证集；`0` 表示不生成验证集），数据划分随机种子 `17`（相同输入和种子得到相同划分）。小批次可能没有 `val/`，不能保证一定产生验证集。数据概览的“原始数采 Episode”是自动统计值，不是可修改参数。导出状态并不等于本地验证已通过。

## 17. Stage 4：本地验证

点击“验证 LeRobot 数据集”。必须看到“本地数据验证通过”，才能进入 HF 上传检查。验证会核对 17 维状态、17 维动作、头部与腕部视频、Task、Episode 边界、来源记录，以及官方 LeRobot API 回读。任何失败都要停止，保留日志并联系工程人员；不要使用命令行强制参数绕过。

## 18. Stage 5：HF Dry Run

切到“Hugging Face”页，确认登录账户与 Private 仓库，再点“检查 HF 上传计划”。Dry Run 只读取当前远端状态并计算计划，**不会上传**。重点看“远端已有 Episode”“本批输入 Episode”“重复 Episode”“新增 Episode”“上传后 Episode 总量”。例如远端已有 100、本批输入 20、重复 5、新增 15，预期总量是 115。若数字与预期不符，暂停并联系工程人员；不要反复点击正式上传试错。

## 19. `NO_NEW_EPISODES` 是什么？

它表示本批数据对应的 Episode 已全部存在于远端，没有新的数据需要上传。不是错误，也不需要手动删除或重新发布。此时正式上传按钮应不可用；记录检查结果即可。

## 20. Stage 6：HF 正式发布

这是**真正修改远端 HF 数据集**的步骤。先仔细核对最新 Dry Run 的仓库、重复数、新增数、预期总量和远端基准 Commit SHA。只有确认确有新增、目标仓库为 Private、本地验证通过时，点击“上传到 Hugging Face”，在弹窗中再次核对并勾选确认。点击后等待任务完成；不要关闭电脑或中断网络，不要同时从另一台机器发布同一仓库。

## 21. 上传时系统做什么？

系统读取当前 HF Dataset → 按来源去重 → 合并新 Episode → 重建完整 LeRobot v2.1 数据集 → 本地验证 → 上传 → 按固定 Commit SHA 重新下载 → 再次回读验证。Chunk/Parquet/MP4 的内部重排由程序负责，现场人员不需要手动处理。

## 22. 上传成功怎么看？

正式发布结果应同时显示：非空 Commit SHA、正确的 Episode 总量、正确的帧数、远端验证 PASS（且对应同一个 Commit SHA）、官方 LeRobot 回读 PASS、文件比对无不一致。仅看到“上传任务结束”或网页上的一条成功提示不够；远端复核失败应视为未完成验收。记录 Commit SHA 和批次名供追溯。

## 23. 什么是 Episode？

Episode 是一次连续的机器人操作轨迹。同一个 Task，例如 `grab the ball`，可以在多次重复实验中产生 Episode 0、1、2；它们是不同的轨迹，不要手工合并。若一条 RAW 轨迹中有无效中间段，导出可能把前后两个有效连续区间分为两个 LeRobot Episode，并保留来源追踪。

## 24. 什么是 Chunk？

Chunk 是 LeRobot 内部组织数据文件的方式，类似把大量帧分块存储。现场人员完全不需要手动管理，也不要重命名、移动或删除远端 Chunk。只需核对 UI 的逻辑 Episode/帧数与验证结果。

## 25. 为什么不是 HDF5？

本项目正式训练数据是 LeRobot v2.1：动作/状态等表格数据在 `data/` 的 Parquet，视频在 `videos/` 的 MP4，任务/统计/来源等元数据在 `meta/`。这就是供下游训练读取的格式；不需要、也不应自行转换成 HDF5。

## 26. “数据处理任务”板块是什么？

它不是机器人 Task 语言指令，而是后台工作记录：数据清洗、LeRobot 导出、本地验证、HF 上传检查或正式上传。可看任务编号、处理阶段、状态、开始/完成时间、耗时和运行日志。状态“等待中”表示已排队，“处理中”表示仍在跑，“已完成（有警告）”需要看原因，“失败/已取消”不能当作成功。技术日志给工程人员排错；普通用户先看中文摘要。

## 27. 浏览器关闭或刷新怎么办？

后台 job 不会因为关闭或刷新网页自动停止，RAW 也不会因此被修改。重新打开 UI，在“处理记录”找到本批任务，点进去查看最新状态和日志。**但关闭 UI 服务终端、关机或断电与关浏览器不同**，可能中断处理；遇到这种情况先找工程人员确认产物状态，避免盲目重跑。

## 28. 常见错误与处理

| 表现 | 常见原因 | 现场处理 |
| --- | --- | --- |
| HF 未登录/认证不可用 | 本机运行账号未 `hf auth login`，或凭据不适用于配置解释器 | 用同一系统账号运行 `hf auth whoami`；仍失败联系管理员 |
| 仓库为 Public、正式上传被阻断 | 目标 Dataset Repo 不符合 Private 要求 | 停止，联系仓库管理员，勿自行换到别的 repo |
| 找不到 Run 或 RAW 路径错误 | 批次目录不在 `/data/vla_runs`、没有 `raw/`，或权限不足 | 核对完整批次目录及交付路径，不改文件内容 |
| 原始 Episode 为 0 | 没有符合命名的 `.npz` 或录制未交付完成 | 暂停，核对数采交付；不要造空 Episode |
| 缺少头部/腕部相机 | 图像目录或相机采集不完整 | 记录 Episode 编号，联系工程人员；不要复制其他帧充数 |
| D3 没有有效数据 | 时间、运动或质量检查排除了全部数据 | 查看中文原因与技术详情，停止导出 |
| `NO_NEW_EPISODES` | 本批已全部存在于远端 | 不需再次上传，记录结果即可 |
| `unknown remote files` | 远端存在程序不认识的文件 | 停止，联系数据工程人员；不要手动删远端文件 |
| `remote HEAD changed` | Dry Run 后远端版本变化 | 重新执行 Dry Run，再核对最新结果；不要沿用旧确认 |

## 29. 哪些问题不要自己修？

现场人员不要改 RAW、Parquet、MP4、来源记录（provenance）、Task 原文、HF Chunk 或远端仓库结构；不要用 `--force` 绕过失败，不要把命令代理值填成实测状态，也不要在不明原因时改门禁。保留 Run 名、Episode 编号、任务编号、截图中的中文摘要及高级原始详情，联系算法/数据工程人员。工程人员负责判断数据是否可修复及如何重新验收。

## 30. 每日标准 SOP

```text
□ 数采完成并确认录制程序已停止
□ 数据完整放入 /data/vla_runs/<run>/raw
□ 启动 UI，选择完整数采批次（不是 raw 或单个 Episode）
□ 核对 Episode 数、Task、头部与腕部相机
□ 开始数据清洗，查看失败/警告 Episode 的中文原因
□ 生成 LeRobot 数据集（一般保持默认划分选项）
□ 本地验证 PASS
□ 确认 HF 登录账户与 Private Dataset Repo
□ 执行 HF Dry Run，核对重复/新增/预期总数
□ 人工确认后才执行 HF Publish；无新增则不上传
□ 远端验证 PASS、官方 LeRobot 回读 PASS、文件比对无差异
□ 记录 Commit SHA、Run 名和最终 Episode/帧数
```

完成以上全部检查才算该批数据发布验收完成。需要算法训练时，由负责 OpenPI/训练的工程人员另行处理，本 UI 不执行训练。
