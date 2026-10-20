# SO-101 · 单黄色螺母 → 棕色盘 · SmolVLA

A reproducible Windows SO-101 / SmolVLA project: 50 LeRobot v3 demonstrations, the selected step-5000 checkpoint, guarded controller v7, and one-click launchers.

## 发布内容

| 内容 | 位置 |
|---|---|
| 模型：第5000步、完整权重与归一化处理器、分词器 | [Hugging Face 模型](https://huggingface.co/wang2065186103/so101-yellow-nut-smolvla) |
| 数据：50条、29,570帧、单腕部RGB、6维状态/动作 | [Hugging Face 数据](https://huggingface.co/datasets/wang2065186103/so101-yellow-nut-to-brown-tray) |
| 代码、一键入口、配置、排坑与复现说明 | 本仓库 |
| GitHub上的完整模型和数据备份 | [本仓库 v1.0.0 Release 附件](https://github.com/damonwangk/so101-yellow-nut-smolvla/releases/tag/v1.0.0)，并附SHA-256清单 |

GitHub主分支存代码和文档，大文件存同一仓库的Release。Release附件是真实模型和数据副本；不只是指向HF的链接，也不依赖Git LFS。HF两个仓库同时保留原生可下载目录及`reproduce.zip`代码包。

## 验证范围

2026-10-20，操作者确认参考机械臂完成了单黄色螺母抓取和落盘任务。当前公开的是这次使用的checkpoint5000与controller v7；这不是统计成功率或新设备泛化结果。

- 原训练在RTX 5060 Laptop 8GB上运行；BF16、batch8、冻结VLM，训练动作专家与相关投影。
- 45条训练 / 5条验证；验证episode：`4,14,24,35,48`，整条隔离。
- 原训练在12000步验证和保存后结束；最低验证loss为5000步约`0.2090`，12000步约`0.2288`。发布5000步，不发布所有优化器状态；完整13个训练检查点仍保留在作者本机。
- 38项控制/准备阶段离线回归、7893周期Guard日志回放、9种姿态730周期归位模拟通过；它们不是物理碰撞验收。
- 发布副本另做数据/严格权重加载/一次离线图像推理检查，结果见`provenance/release-offline-check.json`。

## Windows快速开始

前提：Windows 10/11、Python **3.12**（可用`py -3.12`）、NVIDIA驱动。参考环境：LeRobot0.6.1、PyTorch2.11.0+cu128、transformers5.5.4。不要在Mac运行本控制器。

```powershell
git clone https://github.com/damonwangk/so101-yellow-nut-smolvla.git
cd so101-yellow-nut-smolvla
```

1. 双击`Install.cmd`：创建项目`.venv`并按锁定清单安装依赖。FFmpeg共享版DLL需放入`tools/ffmpeg/bin`或PATH；来源见[FFmpeg官方下载页](https://ffmpeg.org/download.html)。本入口不采集、不训练、不接电机。
2. 双击`DownloadAssets.cmd`：下载公开HF模型和数据。也可从GitHub Release下载两个ZIP，**解压到仓库根目录**；目录前缀已包含在ZIP中。使用`checksums.sha256`核对下载包。程序下载后也按逐文件SHA-256验收；若HF连接不便，可运行`.venv\Scripts\python.exe scripts\download_assets.py --source github`自动下载并核验Release副本。
3. 双击`OfflineCheck.cmd`：检查50条数据和一次GPU推理，串口/电机/相机连接均被拦截。无CUDA时可用`.venv\Scripts\python.exe scripts\release_offline_check.py --skip-model`，此时不会声称模型推理已验证。
4. 在同一台参考设备复现：双击`ConfigureHardware.cmd`，输入你**已校准且已验证映射**的原工作目录，例如作者的`D:\so-101`。它复制配置和校准、记录哈希，不连接电机。其他设备先按`docs/HARDWARE_SETUP.zh-CN.md`准备自己的配置，不能把作者的校准直接刷到别的机械臂。
5. 人员就位并清空回位路径后，双击`TestYellowNut.cmd`。启动程序本身不自动运动，按下文两次Enter流程操作。

`SO101_ROOT`可覆盖工作目录，`SO101_CHECKPOINT`可覆盖本地模型目录；默认从仓库位置解析。相机默认VID:PID=`32E6:9251`、640×480@30FPS；其他相机可设置`SO101_CAMERA_VID`、`SO101_CAMERA_PID`，并保持数据采集时的视角。

## 测试操作

| 阶段 / 按键 | 行为 |
|---|---|
| 加载 | 等待模型、相机和既有配置就绪；不提前显示第二次Enter提示 |
| READY，第一次Enter | 原3秒接入后，受限幅保护回到固定准备姿态 |
| AT_START | 身体误差≤1°并稳定0.5秒，保持力矩；此时再摆放螺母 |
| 第二次Enter | 移开手后确认，等确认后的新相机帧，再开始模型动作 |
| Q | 暂停并保持；下一次Enter仍先归位 |
| R | 主臂跟随手动复位；然后仍按两次Enter流程 |
| S / F | 操作者结果标签并保持 |
| X / Ctrl+C | 先托稳从臂，再退出并释放力矩 |

模型动作阶段限45秒，归位与摆物等待不计入。回位没有环境避障或碰撞规划，先清空路径，AT_START后放物体。当前没有抓取成功检测或自适应力控制，滑动/偏夹后仍可能空载搬运。

固定准备姿态来自训练episode22第0帧，参见`configs/yellow-model-start-reference.jpg`。先固定姿态和物体位置验证，再扩展摆放位置/朝向。

## 控制与数据契约

- 六维：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`；身体单位为度，夹爪为0–100归一化开度。
- **模型action已经是从臂坐标**，不得再加主臂肘/腕补偿。主臂偏移仅用于遥操作及R复位。
- 身体保留原工作边界、5°反馈相对目标限幅、15°/秒、35ms反馈刷新、3秒启动过渡和看门狗。
- 夹爪软件目标只裁剪活动范围，不加入2个百分点误差限幅、独立限速、接触门控或专用平滑；电机驱动自身的力矩/电流保护保持原样。
- 按模型`n_action_steps=50`完成当前预测剩余段后换段，保留最终关键帧；身体限速后，夹爪按同一实际策略时间取样，避免15帧反复重规划截掉释放动作。
- 正式数据`action`是驱动返回的实际从臂目标，`observation.state`是实测状态；不能用视频反推命令。60Hz控制、30FPS数据，主机时间匹配不等于曝光硬件同步。

## 继续采集与重训

`Trial.cmd`单条试采；`Collect50.cmd`连续采集到50条。**新采数据写入`captures/`，不会追加或覆盖发布的`datasets/`。** 连续模式Enter开始、Q保存时主从仍跟随，等待SAVED后复位并开始下一条；复位过程不写入episode。不要连续多次Ctrl+C中断提交。

如需重训，先执行：

```powershell
.venv\Scripts\python.exe scripts\download_assets.py --training-base
```

再双击`Train.cmd`，创建新的时间戳训练目录。复用45/5划分、train-only归一化、batch8、BF16、1e-4学习率、1000步预热、原20000步余弦调度，默认运行12000步并周期验证/保存。每25步打印loss、梯度、lr、显存与耗时。公开的训练参考配置和划分位于`configs/training-reference.json`与`provenance/`。训练loss不等于实机成功率，随机采样/硬件差异也不保证逐位相同结果。

## 目录

```text
scripts/        当前有效控制、采集、训练和发布入口
configs/        作者参考配置、固定准备姿态、参考图
calibration/    作者参考校准，只供理解/离线验算；不可直接刷入其他设备
provenance/     原脚本哈希、45/5划分、归一化统计、验证摘要
models/         下载/解压后生成，Git主分支不包含大文件
datasets/       下载/解压后的原始LeRobot v3发布数据
captures/       后续新采数据，与发布数据隔离
docs/           项目流程、问题与解决方案、其他设备配置说明
```

完整过程与排坑索引见[`docs/PROJECT_GUIDE.zh-CN.md`](docs/PROJECT_GUIDE.zh-CN.md)。源代码、模型、数据按Apache-2.0发布；依赖各自保留上游许可，见`LICENSE`、`NOTICE`。公开内容不含访问令牌、SSH密钥、环境缓存或私人账号配置。
