> 本文记录作者原工作区；完整代码索引可能包含未列入最小发布包的历史审计工具。复现入口以仓库根README为准。

# SO-101 机械臂项目记录

更新：2026-10-20（Asia/Shanghai）。本文记录当前有效配置、实现和验证结论；问题按“原因 → 最终方案”整理，不保留错误堆栈、下载重试和重复日志。

## 目录

1. [当前状态与任务](#1-当前状态与任务)
2. [硬件与软件配置](#2-硬件与软件配置)
3. [校准、坐标与跟随实现](#3-校准坐标与跟随实现)
4. [采集程序与数据格式](#4-采集程序与数据格式)
5. [单黄色螺母数据与验收](#5-当前单黄色螺母任务)
6. [SmolVLA训练、部署与固定起点](#6-smolvla-模型与训练)
7. [关键问题与最终解决方案](#7-关键问题与最终解决方案)
8. [公开归档与他人复现](#8-公开归档与他人复现)
9. [关键代码索引与后续工作](#9-关键代码索引与后续工作)

## 1. 当前状态与任务

- 工作环境：Windows 原生 Python / CUDA / LeRobot，项目根目录 `D:\so-101`；Mac 用于 SSH，并保存本文主文件，不运行机械臂控制。
- 当前任务：**夹取一个黄色螺母，放入前方棕色盘中**。一条 episode 包含一次完整夹取、移动、松开和后撤。
- 已完成：环境安装、主从臂校准与坐标映射、约 60Hz 主从跟随、腕部相机接入、LeRobot v3 采集、SmolVLA 8GB 显存短训练验证。
- 单黄色任务已完成首条试采 `trial/episodes/episode_000000`：1300帧、43.33秒，数据完整且视频确认任务成功；正式50条采集目标已完成，共29570帧、约16分26秒，格式LeRobot v3.0。无未完成提交标记，最后会话正常退出并核验主从臂12电机力矩均为0。正式50条已完成离线验收：37条直接通过、13条带轻微备注通过，视频均可确认放盘成功，无必须剔除条目；原件SHA-256一致。首轮SmolVLA微调按用户确认在12000步完成验证及检查点保存后停止，全部13个检查点保留。
- **2026-10-20，操作员确认当前checkpoint5000 + trajectory-phase-v7完成了单黄色螺母夹取和落盘任务。** 对应参考实机记录为`evaluations/yellow_nut_to_brown_tray/step005000-20261020T094811`。该结论来自操作员现场确认；日志因45秒上限结束，未被当作自动成功判定。尚未开展统计成功率和新设备泛化评估。
- 当前正式连续批次每条由操作员按Enter开始、Q结束并后台保存；保存和复位期间主从保持跟随，用主臂复位，重摆过程不录入数据。托稳从臂后按X退出控制。

## 2. 硬件与软件配置

| 项目 | 当前配置 |
|---|---|
| 主臂 | SO-101 Leader；USB 序列号 `5B7B015114`；当前 `COM4` |
| 从臂 | SO-101 Follower；USB 序列号 `5B7B015028`；当前 `COM3` |
| 相机 | icspring 腕部 UVC 相机；VID:PID `32E6:9251`；DSHOW；当前索引 1 |
| 相机设置 | RGB 640×480，标称 30FPS；相机随从臂腕部运动 |
| GPU / 内存 | RTX 5060 Laptop，8GB 显存；约 64GB 系统内存 |
| Python / 环境 | Python 3.12.15；`D:\so-101\.venv` |
| PyTorch | 2.11.0+cu128；torchvision 0.26.0+cu128 |
| LeRobot | 0.6.1；数据格式 LeRobotDataset v3.0 |
| 模型依赖 | transformers 5.5.4；accelerate 1.15.0；huggingface_hub 1.33.0 |
| 视频依赖 | FFmpeg 8.1.2；PyAV 15.1.0；TorchCodec 0.11.1 |
| 编辑器 | VS Code 1.141.2；`<USER>\AppData\Local\Programs\Microsoft VS Code\Code.exe` |

USB 端口和相机索引可能随重插改变；以序列号和 VID/PID 识别设备。普通启动复用现有校准，不重复设置电机 ID。

安装采用项目内独立 Python 和虚拟环境，CUDA PyTorch 来自官方 cu128 源，LeRobot 与依赖版本见 `requirements-windows.lock.txt`。FFmpeg 的发行包已核对上游校验值。项目内保留环境引导脚本和依赖清单，运行时统一加载：

```powershell
Set-Location D:\so-101
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe --version
```

`environment.ps1` 将模型缓存、uv 缓存、Torch 缓存、临时目录及校准目录统一指向 `D:\so-101`。当前采用 Windows 原生流程，无 ROS / WSL 依赖。

## 3. 校准、坐标与跟随实现

主从臂完成 Windows 校准，校准文件为：

- `D:\so-101\calibration\leader\win_leader.json`
- `D:\so-101\calibration\follower\win_follower.json`

在主臂肩、肘、腕关节上完成编码参考调整，使已使用的工作路径处于连续编码区间；最终采用配置版本 **4**、坐标版本 **3**。当前动作映射集中在 `configs\practical-follow.json`，由 `scripts\follow-practical.py` 的 `WristMapping` 使用。

| 主臂到从臂的映射 | 数值 / 含义 |
|---|---|
| `elbow_flex.pos` | 加 `-90.021978°`，抵消主臂编码参考变化 |
| `wrist_flex.pos` | 加 `-30.197802°`，采用人工对齐姿态得到的近似腕部补偿 |
| 其余通道 | 保持现有校准映射 |

映射的核心代码是先对主臂读数加配置偏移，再做从臂边界与相对目标限幅：

```python
out = {k: float(v) for k, v in action.items()}
for k, delta in self.cfg['offset_degrees'].items():
    out[k] += delta
```

当前控制参数：

- 控制目标 60Hz，已验证运行约 59.9Hz；启动时从从臂当前位置平滑接入 3 秒。
- 前五关节的目标相对实测状态误差上限为 5°，夹爪为 2 个百分比点；这些是目标误差限幅。
- 复用本控制周期从臂反馈，反馈超过 35ms 则刷新；关闭驱动内重复限幅，避免重复读电机。
- 保留从臂绝对边界和编码连续性检查；独立采集与纯跟随不自动归位。模型测试的固定准备姿态及两次Enter流程见6.5。
- 采集准备阶段对小幅超出工作范围的手动起始姿态显示提示。接入首帧保持实测位置，在原3秒接入期内回到既有目标范围；临时起始范围允许前五关节超出最多10°、夹爪超出2个百分点，接入后恢复原反馈范围。总接入余量与每步5°的目标误差限幅独立。超过接入余量仍在启用电机前提示调整姿态，纯跟随入口仍使用原启动检查。连续采集仅在整次控制接入时执行该检查，条目间复位不重新启用电机。
- 启动从首次硬件配置前即承担电机清理责任，配置、目标写入或读回异常均会进入释放及回读核验；启动位置读回和力矩清理使用有限串口重试。检测到残留力矩时，正式窗口在操作员托稳从臂、放稳主臂并输入R后，才调用 `scripts/recover-so101-torque.py` 逐电机释放并核验。恢复不写目标位置或校准，成功后仍等待Enter开始采集。
- 原始 LeRobot 遥操作/采集 CLI 不包含本项目映射，当前机械臂使用项目启动入口。

六维顺序为 `shoulder_pan`、`shoulder_lift`、`elbow_flex`、`wrist_flex`、`wrist_roll`、`gripper`。前五维单位为度，夹爪为百分比点。**训练动作、回放动作和模型输出均为从臂坐标，不能再次施加主臂偏移。**

纯跟随入口是 `D:\so-101\开始主从跟随.cmd`（同 `teleoperate.cmd`）。人工就位后按 Enter 接入；停止前托稳从臂，再按 Ctrl+C。

## 4. 采集程序与数据格式

当前采集复用上述跟随映射和控制参数，由 `scripts\record-so101.py` 实现：

1. 按 USB 身份选择设备并读取当前校准；等待操作员确认就位。
2. 以 60Hz 目标运行控制；相机图像及其时间戳在同一锁内读取，每个观察到的新帧只保存一次。
3. 为图像匹配不晚于该图像主机接收时间的最近从臂实测状态，并记录同一控制周期原生驱动返回的实际目标。
4. 使用单个有界写入队列持久化，写入错误向主线程传递；逐帧时间与映射信息保存在 JSONL。
5. 单条模式Q正常结束后释放并读回扭矩，再等待写入完成、保存episode、关闭数据集并建立逐条查看目录；连续批次Q只结束当前录制并后台保存，跟随持续到操作员按X退出。

实际训练动作来自限幅后驱动返回值，而非未经处理的主臂输入：

```python
safe = ensure_safe_goal_position(goals, cfg['max_relative_target'])
sent = original_send({name + '.pos': float(safe[name]) for name in JOINTS})
```

正常提交使用 `dataset.save_episode()` 和 `dataset.finalize()`；数值 Parquet 在视频编码前关闭。视频使用 H264 / veryfast，写入后的 MP4 继承目标目录读取权限，便于 Windows 播放。

**正式存储为 LeRobotDataset v3.0，不是只有原始图片或 MP4。**

| 文件 / 字段 | 内容 |
|---|---|
| `data\chunk-*\file-*.parquet` | 每帧状态、动作、时间及索引 |
| `videos\observation.images.wrist_camera\chunk-*\file-*.mp4` | 腕部 RGB 视频 |
| `meta\info.json`、`stats.json` | 格式、特征、帧率和统计 |
| `meta\tasks.parquet`、`meta\episodes\` | 任务和 episode 范围、视频偏移 |
| `observation.state` | 六维从臂实测状态 |
| `action` | 六维实际发送的从臂目标 |
| `timestamp`、`frame_index`、`episode_index`、`index` | 固定时间轴和段内/全局索引 |
| `alignment.jsonl` | 逐帧相机、状态、动作主机时间与映射记录 |

LeRobot / MP4 时间轴为 `frame_index / 30`；相机主机接收时间和实际间隔另行保留。标称 30FPS 不代表硬件曝光严格同步，时间对齐以逐帧记录为依据。模型动作的部署时间轴应匹配 30FPS 数据，不能直接以 60Hz 加倍播放。

正式50条连续入口：**Enter录制 → Q结束本条并后台保存 → 等待SAVED/READY → 用主臂复位并重摆 → Enter录下一条**。Q不释放电机，复位过程不录制；保存与下一条写入串行进行，各条帧号和时间轴独立。达到50条后停止录制并保留跟随，**托稳从臂后按X退出控制并释放电机**。Ctrl+C仍会退出控制。

试采及正式单条入口仍为：**托稳从臂 → Q → 等待STOPPED / Saving / SAVED → 关闭窗口**，每次启动保存一条。文件保存成功与任务成功分别确认。

## 5. 当前单黄色螺母任务

```text
D:\so-101\datasets\yellow_nut_to_brown_tray\
  task.json
  采集说明.txt
  trial\
    episodes\
    sessions\
  formal\
    episodes\
    sessions\
    lerobot\
    qc\acceptance_20261019\
```

试采和正式数据独立编号；各自的 `lerobot\` 标准训练目录在首次成功采集时创建。`episodes\episode_000000\` 等是便于查看的逐段目录，训练时使用对应阶段的 `lerobot\`。

任务中文提示为“夹取一个黄色螺母并放入前方的棕色盘中”，数据中统一保存英文指令：

```text
Pick up the yellow nut and place it into the brown tray.
```

- 试采入口：`D:\so-101\开始黄色螺母试采.cmd`（或 `开始数据试采.cmd`）。
- 正式单条入口：`D:\so-101\开始黄色螺母正式采集.cmd`（或 `开始正式采集.cmd`）。
- 正式连续50条入口：`D:\so-101\开始黄色螺母正式采集50条.cmd`，调用 `scripts\record-yellow-batch.ps1 -TargetTotal 50`，再以 `record-so101.py --target-total 50` 运行一个持续控制会话。Q只暂停录制，保存使用后台线程；主从保持跟随，下一条仍由本地Enter开始。
- 出现错误时窗口显示本次原因及摘要位置；单独按Enter保留窗口。尚未启用电机的姿态问题允许调整后输入R重试，输入X退出。正常达到目标后也输入X关闭窗口。
- 暂停后续采使用同一个50条入口。若提示残留力矩，先托稳从臂、放稳主臂，再输入R并回车确认释放；看到RECOVERED后摆好，等待就位提示并按Enter。此恢复只在无其他控制进程时执行，不自动释放尚未由操作员确认的力矩。
- 休息时托稳从臂后按X退出；再次打开50条入口按已提交数量续接。达到50条保存数后不再录制，提示托稳后X退出。`formal\batch-progress.json`记录保存数和控制状态，示范是否成功仍需质量确认。
- 场景只保留一个黄色螺母和目标棕色盘，起始视野能看到螺母、夹爪和盘；保持相机安装与光照一致。
- 初始目标 50 条成功示范：5 个邻近、可达的位置，每个位置 10 条。完成后保留 1–2 秒清晰结果再结束。
- 已完成首条试采：`trial/episodes/episode_000000`，43.33秒、1300帧；视频完整解码，状态/动作/视频及逐帧日志一致，LeRobot读取通过，实测控制59.94Hz。视频确认黄色螺母成功释放于棕色盘。
- 首条检查报告：`D:\so-101\datasets\yellow_nut_to_brown_tray\trial\qc\episode_000000-20261019T085734\audit.json`。正式数据采用独立目录和编号。
- 正式演示统一在放好并后撤后，保持盘和螺母可见1–2秒即结束，避免继续移动使目标离开视野。
- 正式现有 `episode_000000` 至 `episode_000049`，共50条、29570帧；标准元信息与50个逐条目录计数一致，无未完成提交标记。离线验收已完成，50条技术检查通过，视频均可确认释放入盘；37条直接通过、13条带轻微备注通过，无必须剔除条目。原始数据保留。
- 连续采集已在实机使用，正式任务累计保存50条，最后会话 `batch-so101-20261019T102216589258` 正常结束，主从臂力矩释放回读全为0。模拟设备配合真实LeRobot v3临时数据另已验证：连续保存与复位保持控制、复位帧隔离、条目不覆盖、保存故障停机及恢复标记、X退出、启动异常释放与本地R恢复。测试位于 `scripts/tests/`，摘要为 `logs/continuous-recording-validation.json` 和 `logs/resume-recording-validation.json`。

### 5.1 正式50条验收结果

验收代码为 `scripts\audit-yellow-formal.py`：全量对照Parquet、JSONL、元信息及实际发送动作，完整解码全部29570视频帧，并以官方LeRobotDataset在每条首/中/末验证图像、状态及50步动作窗口不跨episode。50条全部通过；未见视频解码错误、近黑帧或相邻完全重复帧。相机实测30.117–30.260FPS、控制59.752–59.948Hz，最大相机主机接收间隔78.56ms。验证保存记录与主机时间关系，不等同于硬件曝光严格同步证明。

视频内容按每条全长2FPS分镜逐条审阅（2039张、94页），11条相关片段再以5FPS复核。时间点为约值，未逐一视觉判断全部30FPS原帧。可确认50条均将黄色螺母释放在棕色盘内；未见需要剔除的明显任务失败或人工补放。

- 对位较慢：`000000`、`000001`、`000010`，保留完整演示。
- 成功后停留较长：`000003`，可选保留至约22秒。
- 成功后的尾段目标裁切或出画：`000002`、`000018`、`000030`、`000031`、`000034`、`000039`、`000040`、`000043`、`000044`。已标记约时间点及可选尾段截止时间，本次未裁剪。

结果目录：`D:\so-101\datasets\yellow_nut_to_brown_tray\formal\qc\acceptance_20261019`。`review.html`可按条目筛选、查看原视频并跳到证据时间；`episodes_review.csv`列出50条结果，`quality_labels.json`保存独立标签，`train_include.json`为可用episode名单，QC名单本身不划分训练/验证；正式训练45/5划分另见6.1。全程保留原始`lerobot/`、`episodes/`、`sessions/`，539个文件路径的验收前后SHA-256一致，清单保存在同目录。

本机验收报告副本：`<MAC_PROJECT_RECORDS>/数据验收报告.md`。当前场景相近，尚未量化起始位置覆盖均匀性或评估模型泛化。验收未启动机械臂、采集、动作重放或训练。

## 6. SmolVLA 模型与训练

采用官方 `lerobot/smolvla_base` 预训练权重进行任务微调。官方权重及所需配置、分词器保存在：

- `D:\so-101\models\smolvla_base`
- `D:\so-101\models\SmolVLM2-500M-Video-Instruct`（模型配置、processor 与 tokenizer）

已使用两条真实试采完成一次 **100 步训练可运行性验证**：

| 项目 | 实测结果 |
|---|---|
| GPU / batch | RTX 5060 Laptop 8GB；batch 1 |
| 精度 / 训练范围 | BF16 混合精度；冻结 VLM，训练动作专家及投影层 |
| 总参数 / 可训练参数 | 450,046,176 / 99,880,992 |
| 输入 / 动作窗口 | 单腕部图像，填充缩放为 512×512；50 个动作步 |
| PyTorch 显存峰值 | 分配 1.940GiB；保留 2.061GiB，不含全部驱动和桌面开销 |
| 训练更新耗时 | 稳定阶段每步中位约 0.430 秒 |
| 完成标准 | 100 次更新、有限损失与梯度、权重变化、检查点保存及读取一致性均通过 |

该短跑结果验证本机能运行上述微调配置；正式模型随后使用单黄色螺母示范完成训练并在参考设备完成任务，尚未量化成功率。短跑检查点不作为正式任务模型，当前只保留可复用的训练验证摘要与官方预训练权重。

训练实现位于 `scripts\train-smolvla-8gb-smoke.py`，复用官方 LeRobot 数据读取器、SmolVLA、预处理器、优化器、训练循环和检查点保存。图像键及统计根据本任务数据配置，动作保持从臂坐标，不加入主臂补偿。脚本已默认读取单黄色正式数据，目录为空时直接提示，不会自动采集。

短跑入口仍默认100步，用于可运行性验证。正式任务已另建训练入口，使用按episode隔离的训练/验证名单、学习率预热与调度、周期检查点和断点续训。部署接入已完成；统计成功率评估尚未进行。

后续有已保存数据时，可用以下命令再次验证训练配置（本次未执行）：

```powershell
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe .\scripts\train-smolvla-8gb-smoke.py --steps 100 --batch-size 1
```

当前成功验证摘要保存在 `D:\so-101\logs\smolvla-8gb-training-summary.json`。

### 6.1 单黄色任务正式训练

首轮 `yellow_nut_smolvla_20261019T110537` 已按用户要求提前结束。45条训练示范共26722帧，验证集为`000004、000014、000024、000035、000048`共5条2848帧，每10条采集记录中保留1条，整条隔离。首轮使用全部已验收示范，不应用可选尾段裁剪。状态和动作归一化统计仅用45条训练数据计算，保存在日志目录，不修改数据集原件。

| 参数 | 正式首轮 |
|---|---|
| 优化步数 / batch | 原预算20000步，实际在12000步验证及保存后停止；batch 8，无梯度累积 |
| 精度 / 训练范围 | BF16 autocast；冻结VLM，训练动作专家及状态/动作相关投影 |
| 学习率 | 1e-4；1000步预热；余弦衰减至2.5e-6 |
| 输入 / 输出 | 单腕部相机；模型内部填充512×512；状态6维，动作窗口50×6 |
| 保存 / 验证 | 首次100步保存；此后每1000步和最终步保存，每1000步验证完整5条保留示范 |
| 数据读取 | 官方LeRobotDataset，PyAV，num_workers=0，seed=1000 |
| 权重 / 上传 | 从本地官方smolvla_base完整权重微调，Hub/WandB上传关闭 |

训练入口为`scripts\train-yellow-smolvla.py`，由`scripts\run-yellow-smolvla-formal.ps1`加载现有环境并启动，实际更新、验证和完整检查点保存复用安装的官方LeRobot训练循环。Windows任务`SO101-SmolVLA-Yellow-Formal`在登录会话中后台执行，独立于SSH和VS Code窗口。模型、处理器、优化器、调度器及随机状态随完整检查点保存；训练显存峰值约2.56GiB（PyTorch分配，不含全部桌面/驱动开销）。

2026-10-19 13:37按要求在第12000步验证及完整检查点保存后停止。最终验证loss为0.2288，最低验证loss为第5000步的0.2090；后续验证未改善，因此不强行跑满原预算。保留首次100步及1000至12000步共13个完整检查点，不修改其原配置或学习率调度。`early_stop_request.json`、`early_stop_result.json`记录停止条件与结果，状态为`stopped_early`；训练入口会阻止误续训该已结束批次。此结果尚不能代表实机成功率。

- 本轮日志：`D:\so-101\logs\yellow_nut_smolvla_20261019T110537`。`run_spec.json`记录配置，`split.json`记录名单，`training_normalization_stats.json`为训练集统计，`training_status.json`为当前状态，`metrics.jsonl`为训练/验证/保存事件。
- 模型输出：`D:\so-101\outputs\train\yellow_nut_smolvla_20261019T110537`。完整编号检查点位于`checkpoints/`，普通文件`last_checkpoint.json`记录最近检查点路径，无Windows符号链接权限依赖。`best_checkpoint.json`记录保留集损失最低的已保存检查点。
- 续训入口：`D:\so-101\继续黄色螺母正式训练.cmd`。运行中的批次可从完整检查点恢复优化器、调度器、随机状态和采样进度；本轮已按要求结束，防误续训保护会直接提示。需要进一步训练时另行明确配置，单进程锁防止重复训练。
- VS Code已打开训练代码、启动脚本、配置、划分、实时状态/日志及`D:\so-101\黄色螺母训练流程.md`，后者包含数据→预处理→模型→优化→验证→保存的流程及代码索引。

实时查看采用只读文本页：`D:\so-101\logs\yellow_nut_smolvla_20261019T110537\实时训练指标.log`。`scripts\write-yellow-training-live-view.py`每秒读取既有状态/指标并刷新该文件，VS Code自动加载外部变化；显示步数、loss、梯度范数、学习率、显存、验证结果和最近检查点。训练指标仍按原每25步记录，验证/保存期间步数暂停；关闭文本页不影响训练。此查看方式保留VS Code受限模式，不更改工作区信任。

下次运行已改为训练代码直接打印：`train-yellow-smolvla.py`中的`monitored_update`在前5步及之后每25步调用`print_training_event`，逐行显示`step/total、loss、grad_norm、lr、gpu_peak、update_s`并立即flush；验证、检查点保存、完成和异常结束也打印提示。`ConsoleTee`将控制台输出同步保存到`stdout-*.log`、`stderr-*.log`，保留原有`metrics.jsonl`与状态文件。启动脚本直接运行Python，控制台可实时查看，无需另开日志生成程序。已通过Windows Python编译、PowerShell语法及控制台/文件双输出检查；本轮进程未重启，改动从下次启动生效。

20000步是原始首轮预算，batch 8的样本量不同于官方示例batch 64；实际根据保留集表现提前结束。保留集flow matching损失不等于真实抓放成功率。第5000步模型离线动作检查已完成，统计实机成功率尚未评估；参考设备任务完成已由操作员确认。

### 6.2 第5000步模型离线检查

已完成官方严格加载、500个权重张量有限值检查、45条训练示范归一化统计核对。5条保留示范各抽20个观测，两个随机种子共200段50×6预测，60000个输出均无NaN/Inf；第5000步检查点全部文件检查前后SHA-256一致。推理不输入真实动作标签，原始数据只读，不连接硬件。

首个动作的身体五关节平均绝对误差1.73°，夹爪1.05个百分点；完整50步身体平均误差3.66°，保持当前姿势基线为5.65°。短时精度仍需关注：首个动作基线误差0.94°，低于模型。35/200次首目标越过既有工作边界，83/200次需单次相对限幅；肩抬升最大越界10.24°，去除启动阶段后仍为7.51°。模型文件与数值链路通过，原始动作尚不能直接发送给电机。

离线推理显存峰值约1413MiB；批量推理耗时不代表实机控制延迟。详细记录见`第5000步模型离线检查.md`，Windows结果位于本轮日志的`offline_check_005000`，含`summary.json`、`sample_flags.csv`、`predictions.npz`及检查点哈希/清单。后续控制接入见6.3，统计实机成功率尚未评估；参考设备任务完成已由操作员确认。

### 6.3 第5000步模型实机测试接入

当前版本为`trajectory-phase-v7`，入口仍为桌面“黄色螺母模型5000测试”及`scripts\start-yellow-model-test.ps1`。模型、数据、原校准与`practical-follow.json`保持原样，模型目标直接采用从臂坐标；主臂偏移只用于R复位跟随。v7增加6.5中的固定准备姿态，继承完整50步段执行与策略时间对齐。

按用户要求，当前模型测试程序的夹爪仅保留由原校准生成的绝对活动范围：取消相对反馈2个百分点、15个百分点/秒速率上限、夹爪专用平滑/重叠渐变和启动渐变，不设置接触到位判定。`Guard.command`将夹爪目标直接裁剪到该范围；夹住物体后目标仍可继续向闭合方向请求，不因实际开度达不到空载目标而冻结整臂。该设置也用于本测试程序的R复位，不修改已验收的独立采集/跟随程序。

五个身体关节仍保留原绝对工作边界、编码器连续性检查、3秒启动过渡、35ms反馈刷新、单次相对反馈5°以及15°/秒上限。身体共同缩放比例独立决定策略进度，最终原生相对限幅检查只应用于身体；底层驱动`max_relative_target=None`，没有重复的夹爪限幅。原有小幅起始摆位恢复保持不变。

`scripts\yellow_action_queue.py`按检查点的`n_action_steps`设置执行段；当前检查点`chunk_size=50`、`n_action_steps=50`。异步推理继续更新暂存预测，但须完成当前预测的剩余段至最后一帧，才接入下一包；请求期间按实际策略进度对齐，分数偏移也保留第49号最终关键帧。7帧三角平滑与5帧换段过渡仅作用于身体，前段已耗尽时以末姿态作为过渡参考。

60Hz伺服每周期最多提议半个30FPS策略帧，身体五关节限幅计算实际可推进比例，再按同一个已接受策略时间点取夹爪目标；夹爪仅裁剪到活动范围，不添加速率、相对误差、接触或专用平滑限制，也不参与身体缩放计算。身体受限时策略时间共同减慢，夹爪受阻不会冻结整臂。相机、推理超时及3秒不能推进一个策略帧时仍保持。完成一段后才换预测，会降低换段频率；Q、故障与超时仍可立即中断。

第一次本地Enter执行原3秒接入并移动到固定准备姿态，显示`AT_START`后保持，摆好螺母后第二次Enter才启动模型。45秒上限从正式策略阶段计算，不包含归位与摆物等待。Q保持、R主臂复位、S/F人工结果标记、托稳从臂后X/Ctrl+C退出及回读力矩释放均继续保留。Q、复位或故障会取消准备阶段并清空旧段及暂存预测，旧轮次结果不混入新测试。

当前版本通过38项离线回归、三段历史日志共7893个真实`Guard.command`集成模拟周期、9种起始姿态共730周期归位模拟及300周期夹爪受阻模拟。身体工作边界、单次相对反馈5°、每周期不超过0.25°及原速度限制继续有效；夹爪仍可超过原相对误差与速度限制但始终在活动范围内。三段策略回放均无提前换段、队列断流或看门狗超时，均可执行至第49号索引。串口打开、电机总线连接及相机连接调用被显式拦截，调用次数均为0。报告：`logs\yellow-fixed-start-v7-regression.json`，入口为`scripts\verify-yellow-fixed-start-v7.py`。这些是模拟反馈下的控制与启动时序验证，不代表碰撞路径验证或实机抓放成功。

程序会在独立测试目录`evaluations\yellow_nut_to_brown_tray\step005000-时间\input_images`留存每次推理请求图像的JPEG副本，`control.jsonl`的预测事件包含对应`input_image`、`input_state`、时间及动作。图像由推理进程保存，写入异常仅记录诊断错误；不写入正式数据、不自动转成示范。这些是推理请求快照，不是30FPS测试视频。代码为`scripts\test-yellow-smolvla-robot.py`；修改前v6已归档到`archive\model-test-before-fixed-start-v7`。当前版本已部署，通过离线验证后由操作员实测并确认任务成功；以后从原入口启动仍等待本地Enter开始。本次归档不启动控制。

### 6.4 抓取与落盘调试

本轮重点检查对准、接近、闭合、抬升和放盘的连续性，结合腕部输入快照、关节状态及发送目标定位偏差。

已只读比较50条示范起始图与准备图，准备阶段可见黄色螺母；2026-10-20 08:55:59测试起始夹爪开度约54，接近示范中位数约52。起始对比资料位于`logs\yellow-grasp-input-review`，程序为`scripts\inspect-yellow-grasp-input.py`。

本次视频`IMG_4478.MOV`与测试目录`evaluations\yellow_nut_to_brown_tray\step005000-20261020T085559`对照显示：约13–14秒夹起，约25秒在托盘内侧靠盘沿处发生脱离；掉落前后存在短暂松夹，发送目标由约33–34增大到37.58，实际开度最高36.50，随后重新收紧。夹爪开度单位为百分比点。视频与日志按文件时间近似对齐，不用于毫秒级先后判定；不能仅据此排除抓持偏浅或接触盘沿的影响。此时仍在模型控制阶段，未触发45秒上限。

收尾命令核查覆盖全部50条正式数据，其中45条实际训练示范中44条满足“持续闭合后连续6帧开度大于45”的粗略松夹判据，后续保留约4.3–9.5秒；另1条未达到该阈值，不据此判为数据错误。抽看8条代表性释放/尾帧可见落盘与夹爪离开的收尾内容。脚本为`scripts\inspect-yellow-release-endings.py`，结果为`logs\yellow-release-review`。该检查说明收尾数据存在，不等于模型已学会稳定执行。

离线核对本次157段预测、55次换段及2629个模型控制周期，夹爪命令可由保存的预测和队列进度复原，最大差异小于`3e-13`个百分点。旧15帧窗口最高只用到第29号源索引；11段已接入预测的张开动作位于该段被替换前尚未执行的后部。另发现夹爪使用身体限速前的提议时间点，与实际接受时间点的开度差最大约1.10个百分点，以及分数延迟对齐遗漏最后一个关键帧的问题。原始审计为`logs\yellow-replanning-audit-v5.json`，脚本为`scripts\audit-yellow-replanning.py`。

6.3中的完整段执行、最终关键帧保留与时间对齐修复已完成。用同一批保存预测及到达时间、重映射请求进度并模拟反馈进行对照，旧15帧版本闭合后无大于45的张开目标，新版本可执行至第49号索引，闭合后最大张开目标约47.99；身体限制继续有效。该结果只证明后段动作不再被窗口持续截掉，不证明现场放置高度、位置和抓持稳定性已解决。修复后采用固定准备姿态并由操作员实测确认任务成功；扩展位置与朝向前再决定是否补采，不因单次表现直接增加训练步数。

### 6.5 固定起始姿态与两次Enter

从45条实际训练示范的第0帧中筛选出43个边界内、开爪状态的候选，再按各维稳健尺度归一化的平均L1距离选取代表点。固定姿态采用`episode_000022`第0帧的真实`observation.state`，未使用5条验证示范、未拼接各关节中位数、未改原数据。来源文件及SHA-256写入配置，参考图也来自同一帧。

| 关节 | 准备目标 |
|---|---:|
| shoulder_pan | -5.2747° |
| shoulder_lift | -77.0989° |
| elbow_flex | -0.4835° |
| wrist_flex | 87.5604° |
| wrist_roll | -1.8901° |
| gripper | 52.1472个百分点 |

配置为`configs\yellow-model-start-pose.json`，保留原始浮点精度；参考图为`configs\yellow-model-start-reference.jpg`。选择脚本为`scripts\select-yellow-start-pose.py`，只读统计在`logs\yellow-start-pose-review\summary.json`。45条训练起始姿态中间80%的底座角度约为-7.00至-2.55°、肩抬升-77.33至-73.51°，夹爪48.19至59.51个百分点；起始变化不意味着数据无效，统一姿态用于减少当前测试变量。

操作流程：打开原入口后等待`READY`，第一次Enter启用原接入流程并受原Guard限制回到准备姿态；五个身体关节实测误差均不超过1°且持续0.5秒后显示`AT_START`，保持力矩等待摆物；参照图摆好螺母并移开手后第二次Enter确认，收到确认后的新相机帧才开始正式策略。每轮Q/S/F后再次Enter仍先归位。R保留主臂手动复位。夹爪是否达到空载目标不参与归位到位判定，也不阻塞身体或策略进度。

终端按阶段提示：加载完成并进入`READY`后显示“模型加载完成。请按一次 Enter 复位到固定起始位置。”；接入时显示“正在复位，请稍候……”；到达`AT_START`后才显示“复位完成。请摆好黄色螺母并移开手，然后按第二次 Enter 开始推理。”。窗口标题不预告第二次Enter，画面按当前阶段显示对应提示；启动脚本提示等待加载并遵循后续提示。此项仅调整显示文字，控制流程不变，已检查语法及显示代码之外的结构一致性，未启动程序或机械臂。

准备阶段同样保留相机和推理进程失联保护；身体3秒未接近目标约1°或超过按距离计算的准备超时则进入HOLD。归位是关节目标运动，不包含环境碰撞检测或避障；第一次Enter前须留空运动路径，螺母在`AT_START`后摆放。该姿态来自实际示范且数值在界内，参考设备已按此流程实测并由操作员确认任务成功；其他摆位或环境仍需单独确认路径。

滑动、接触角度和闭合目标会共同影响夹持，固定起始姿态不等于解决抓持或识别空夹。当前驱动使用位置模式，现有图像/位置日志不能测得实际夹持力。已核对本机及[官方v0.6.1驱动](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py)：配置逻辑保留夹爪50%最大力矩、保护电流及过载保护。这些电机保护与已取消的目标相对误差/速率限制不同，本轮未修改或实测寄存器，不据此认定力矩不足。参考设备固定初态和物体摆放后已由操作员确认完整任务成功；如需扩展位置/朝向，再收集有针对性的成功与纠偏示范，本次归档不新增采集或训练。

## 7. 关键问题与最终解决方案

### 7.1 坐标、起点与机械接触

| 问题 | 原因与最终方案 | 实现位置 |
|---|---|---|
| 主从姿态对不上、编码读数在工作段跳变 | 校准后调整编码参考，使常用路径连续；主臂肘/腕仅在遥操作映射补偿。模型输出本来就是从臂坐标，不能重复补偿 | `practical-follow.json`、`WristMapping` |
| 重插设备后端口/相机索引变化 | 用USB序列号和VID/PID定位；复用已验证的校准，不以索引当设备身份 | `follow-practical.py`、`record-so101.py` |
| 示范起始姿态不一致，部署初态难重复 | 只从45条训练示范选取真实代表点episode22第0帧，第一次Enter归位，稳定后再显示第二次Enter，使用确认后的新相机帧推理 | `yellow_start_pose.py`、起点JSON/参考图 |
| 接触时螺母滑动、不同角度闭合不足 | 软件目标角度不等于夹持力；保持夹爪活动范围，保留底层电流/力矩保护，不增加空载到位门控。固定初态可减少变量，仍不能代替位置/朝向覆盖和纠偏示范 | `Guard.command`；必要时补采 |

### 7.2 连续采集、提交与恢复

| 问题 | 原因与最终方案 | 实现位置 |
|---|---|---|
| Q结束一条后电机释放、下一条复位困难 | 单条入口会退出控制；连续50条改为一个持久控制会话，Q只停录并后台保存，主臂复位过程不录入，SAVED后下一次Enter开始 | `record-so101.py`、`record-yellow-batch.ps1` |
| 暂停后续采、28条后重新启动不顺畅 | 以已提交episode数量续接，保存与采集串行；启动异常清理扭矩，有限重试。残留扭矩仅在操作员托稳并输入R后释放，不自动写位置 | `recover-so101-torque.py`、batch-progress |
| 视频/条目尚未出现或保存被中断 | Q后等待SAVED/finalize，禁止连续多次Ctrl+C打断提交；用未完成标记和已提交计数区分完整数据，避免覆盖原episode | `EpisodeWriter`、`write_episode_view` |
| Windows MP4显示无访问权限 | 关闭数值/编码写入资源并使输出继承目标目录读取权限，完成后可在普通播放程序中打开 | 视频提交逻辑 |
| 只有视频不足以训练 | 在线同步保存图像、实测state、实际发送action和各流主机时间；标准输出LeRobot v3，JSONL保留对齐证据，不事后从视频估计action | `select_control_sample`、LeRobot写入器 |

### 7.3 8GB训练与部署动作时序

| 问题 | 原因与最终方案 | 实现位置 |
|---|---|---|
| 8GB能否微调、loss如何实时看 | BF16 + 冻结VLM，batch8实际完成12000步；训练循环每25步flush打印并同步日志。按保留集选5000步，完整13检查点仍本地保留 | `train-yellow-smolvla.py` |
| 数据泄漏或验证虚低 | 整条episode隔离为45/5；归一化只从训练45条统计，验证不参与。loss只用于模型比较，不当作实机成功率 | `split.json`、训练集统计 |
| 身体限速后动作队列反复从开头接入 | 用已接受的实际策略进度衔接异步预测，30FPS模型时间与60Hz伺服分开，减速时推进策略时间而非丢动作 | `yellow_action_queue.py` |
| 后段松爪没执行，15帧重规划一直截断尾段 | 当前预测按检查点50步剩余段执行到最终关键帧，再接下一包；分数延迟对齐也保留最后一帧 | `RetimedActionQueue` / 队列衔接 |
| 夹住物体后整臂停住 | 夹爪无法达到空载闭合目标不应参与身体共同缩放；身体独立决定进度，夹爪仅裁剪活动范围 | `Guard.command` |
| 限速后夹爪开合与身体错相 | 先计算身体可接受进度，再按同一时间点插值夹爪，避免提前执行松夹目标 | Guard与队列阶段提交 |

以上为已采用方案，保留原始数据、模型和已验证的身体边界/5°单步限幅。38项离线回归验证数值和时序，参考设备成功来自操作员实测；两者不能互相替代。

## 8. 公开归档与他人复现

发布账号与可见性由用户确认：Hugging Face `wang2065186103`、GitHub `damonwangk`，均公开。发布副本位于Windows `D:\so-101\releases\yellow-nut-smolvla-v1`，不改动原工作代码、数据、校准和全部13个训练检查点。

| 归档 | 内容 / 位置 |
|---|---|
| [HF模型](https://huggingface.co/wang2065186103/so101-yellow-nut-smolvla) | 5000步完整推理权重、前后处理与归一化张量、tokenizer、模型卡、复现代码包 |
| [HF数据](https://huggingface.co/datasets/wang2065186103/so101-yellow-nut-to-brown-tray) | 训练实际使用的50条原生LeRobot v3数据，29570帧；原153文件不改写，另附数据卡与代码包 |
| [GitHub](https://github.com/damonwangk/so101-yellow-nut-smolvla) | 同一仓库保存代码/README/过程文档，并在v1.0.0 Release保存实际模型、数据ZIP与SHA-256清单，避免Git LFS额外依赖 |

可移植副本从所在仓库解析路径；仅替换原机器绝对路径和发布元数据，不修改模型张量、数据、动作算法或校准参考值。训练45/5划分、原始脚本哈希、回归摘要与发布SHA-256清单保留在`provenance/`。模型权重和归一化张量逐文件对照原件，数据153文件逐文件对照原件；复现程序下载后再次验证哈希。

复现入口：`Install.cmd` → `DownloadAssets.cmd` → `OfflineCheck.cmd` → `ConfigureHardware.cmd` → `TestYellowNut.cmd`。离线检查已通过：50条/29570帧可读、严格模型加载及一次50×6预测成功，串口/总线/相机连接计数均为0。安装入口按参考锁定依赖生成，未在另一台全新Windows上从零实测。

作者校准只作参考，不可直接应用到另一台机械臂。ConfigureHardware导入使用者已经校准并验证过的工作目录并记录哈希，不运行控制；其他设备的坐标范围、USB身份与起点必须适配。实机仍采用本地两次Enter；新采数据写入`captures/`，与下载数据隔离。训练入口新建时间戳目录，不覆盖参考检查点。

**2026-10-20公开发布完成。** 采用Mac中转和已登录网页上传HF、Mac已有GitHub CLI登录发布；公司Windows未登录发布账号、未保存发布令牌。Mac中转副本与Windows发布副本逐文件SHA-256一致，未在Mac安装或运行机械臂环境。

| 验收对象 | 固定原生文件版本 / 结果 |
|---|---|
| HF模型 | `8fac233eacca92e7e4cff8988bf03bcdc313b6e9`；18个推理/分词器文件全部核对大小与服务端LFS SHA-256或Git对象哈希，另有模型卡、许可和复现代码包 |
| HF数据 | `509728a12d1dec22d258254bb2cf614867b1b8b9`；153个原生文件、201730299字节全部核对大小与服务端哈希，另有数据卡、许可和复现代码包 |
| GitHub | 公开主分支代码已逐文件核对Git对象内容；`v1.0.0` Release保存模型ZIP（1202671356字节）、数据ZIP（201770237字节）、复现代码ZIP和`checksums.sha256`；附件服务端SHA-256与本地一致 |

`release.json`固定上述模型/数据版本；复现包的后续文档提交不改动这些原生文件。发布校验依据在`provenance/publication-validation.json`；大文件校验清单见[GitHub Release](https://github.com/damonwangk/so101-yellow-nut-smolvla/releases/tag/v1.0.0)。公开内容不包含访问令牌、SSH密钥或环境缓存。原工作区代码、校准、全部原始数据和13个训练检查点继续保留。

## 9. 关键代码索引与后续工作

| 文件（均位于 `D:\so-101`） | 重点 |
|---|---|
| `scripts\environment.ps1` | Python、依赖、缓存、FFmpeg 路径 |
| `configs\practical-follow.json` | 当前控制频率、坐标偏移、限幅、校准绑定 |
| `scripts\follow-practical.py` | `WristMapping`、设备建立与跟随接入 |
| `scripts\record-so101.ps1` | 当前任务、trial/formal 路由及单条启动入口 |
| `scripts\record-yellow-batch.ps1` | 正式50条持续控制入口、已提交数量计数、启动重试与续接 |
| `scripts\recover-so101-torque.py` | 本地操作员确认后的残留力矩释放及12电机回读核验，不写目标或校准 |
| `scripts\record-so101.py` | `read_camera_frame`、`select_control_sample`、`EpisodeWriter`、`open_dataset`、`write_episode_view`、Q 停止与提交 |
| `scripts\train-smolvla-8gb-smoke.py` | 100步低显存可运行性验证 |
| `scripts\train-yellow-smolvla.py` | 正式微调、45/5划分、训练集统计、验证与断点恢复；`monitored_update`循环打印，`print_training_event`格式化，`ConsoleTee`同步存日志 |
| `scripts\run-yellow-smolvla-formal.ps1` | 训练启动、实时控制台输出与退出状态 |
| `scripts\write-yellow-training-live-view.py` | 只读训练指标，实时刷新供VS Code查看的文本页 |
| `scripts\stop-yellow-after-12000.py` | 确认12000步验证完成、完整检查点及指针齐全后停止；保留全部检查点 |
| `scripts\check-yellow-best-offline.py` | 第5000步模型严格加载、数值与统计核对、保留集抽样动作推理；不连接硬件 |
| `scripts\test-yellow-smolvla-robot.py` | 第5000步实机测试；身体保护、夹爪仅活动范围、异步推理与请求图像留存、暂停及退出清理 |
| `scripts\yellow_action_queue.py` | 实际目标进度衔接、50步缓冲、身体五关节共同减速与夹爪策略插值 |
| `scripts\yellow_start_pose.py` | 真实准备姿态配置验证、实测到位稳定、两次Enter及新帧门控，不连接硬件 |
| `configs\yellow-model-start-pose.json` | 来自训练episode22第0帧的固定起始姿态与来源哈希 |
| `configs\yellow-model-start-reference.jpg` | 同一帧的腕部参考画面，用于到位后摆放螺母 |
| `scripts\select-yellow-start-pose.py` | 只读选择训练起始姿态代表点并提取同帧参考图 |
| `scripts\test_yellow_start_pose.py` | 准备阶段、二次确认、新图像及真实Guard归位回归 |
| `scripts\verify-yellow-fixed-start-v7.py` | 当前版本38项回归与日志/归位集成验收，显式拦截硬件 |
| `scripts\audit-yellow-replanning.py` | 只读复原原始15帧日志与不同窗口的固定预测回放 |
| `scripts\test_yellow_replanning.py` | 尾部动作、最终关键帧、真实Guard策略时间对齐回归 |
| `scripts\verify-yellow-replanning-v6.py` | 28项回归、三段日志与真实限幅集成验证，显式拦截硬件连接 |
| `scripts\inspect-yellow-release-endings.py` | 只读统计示范收尾命令并提取代表性尾帧，不连接硬件 |
| `scripts\test_yellow_action_queue.py` | 队列及协同限速回归、保存预测时序回放 |
| `scripts\start-yellow-model-test.ps1` | 本地可见测试窗口与重复控制进程防护 |
| `scripts\audit-so101-episode-current.py` | 已保存 episode 的离线检查，不控制机械臂 |
| `scripts\audit-yellow-formal.py` | 正式50条全量完整性、视频解码、官方读取与SHA-256检查；QC结果独立存放 |
| `datasets\yellow_nut_to_brown_tray\task.json` | 单颗任务说明、成功条件与采集计划 |

VS Code 已在 Windows 登录桌面打开整个 `D:\so-101` 和上述主要代码；项目记录副本已打开为编辑器标签页。控制、采集和训练均由操作员明确启动。

首轮已在12000步保存后结束，第5000步模型离线检查及带限幅的实机测试接入已完成。完整段执行、策略时间对齐和固定准备姿态均已接入并通过离线验证，身体保护与夹爪仅活动范围设置保留。操作员已确认固定起点后的完整任务成功；下一步可在固定条件下进行约20次独立抓放评估，之后扩展位置和朝向覆盖。当前不启动机械臂；以后第一次本地Enter归位，第二次Enter开始模型。

后续每完成环境、控制、采集或模型阶段，更新本文对应章节的最终参数、代码位置、验证结论和下一步；覆盖过时信息，不追加错误堆栈、下载重试、失败方案或大量逐帧日志。本文主文件保存在用户当前电脑 `<MAC_PROJECT_RECORDS>/机械臂项目记录.md`；Windows 工程内仅保留同内容副本，供 VS Code 查看。

官方参考：[SO-101](https://huggingface.co/docs/lerobot/so101)、[SmolVLA](https://huggingface.co/docs/lerobot/v0.6.1/en/smolvla)、[LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)。
