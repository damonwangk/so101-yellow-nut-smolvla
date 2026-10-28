<a name="sources"></a>

# 20｜代码阅读地图、文件组合与来源

[教程目录](../README.md) · [上一章](19-reproduce.md)

> **学习目标：建立可追溯的工程习惯**
>
> 能从一条结论回到数据、代码、配置或验证记录。

## 先读哪几段代码

| 顺序 | 文件 / 函数 | 要理解的问题 |
|---|---|---|
|1|`record-so101.py`：read_camera_frame、select_control_sample、EpisodeWriter、EpisodeBatch|图像怎样匹配状态/动作，Q如何保存而不断控制|
|2|`train-yellow-smolvla.py`：split_with_train_stats、monitored_update、verified_save|45/5、归一化、日志、检查点怎样接入官方trainer|
|3|官方`dataset_reader.py`：_get_query_indices|50步窗口与episode末尾padding|
|4|官方`processor_smolvla.py`、`modeling_smolvla.py`|预处理、前后缀、loss、完整块预测|
|5|官方`smolvlm_with_expert.py`、`flow_matching.py`|冻结范围、注意力、10次Euler积分|
|6|`test-yellow-smolvla-robot.py`：worker、Guard.command、main|图像到预测、异步通信、保护与状态机|
|7|`yellow_action_queue.py`：offer、target、gripper_target、advance|请求进度、完整段、身体/夹爪相位|
|8|`yellow_start_pose.py`：FixedStart.begin/confirm/update|首次归位、稳定判定、第二次Enter和新帧|

GitHub版提供固定源码快照和行号链接，离线交互包保留原网页代码阅读器。`project/`来自已公开便携版本，已把原机器绝对路径改为根目录解析，动作算法与参考配置保留；`lerobot/`来自本次Windows已安装官方0.6.1源码，另补同tag的`vla_utils.py`辅助函数。它们是阅读附件，Mac没有安装或运行机械臂环境。完整可执行复现包仍以上述公开仓库为准，不能只拿阅读附件当安装包。

### 完整带行号代码快照

| 来源 | 文件 |
| --- | --- |
| project | [configure_hardware.py](../code/project/configure_hardware.py) |
| project | [download_assets.py](../code/project/download_assets.py) |
| project | [environment.ps1](../code/project/environment.ps1) |
| project | [explain_real_batch.py](../code/project/explain_real_batch.py) |
| project | [explain_saved_timing.py](../code/project/explain_saved_timing.py) |
| project | [follow-practical.py](../code/project/follow-practical.py) |
| project | [practical-follow.json](../code/project/practical-follow.json) |
| project | [prepare_training.py](../code/project/prepare_training.py) |
| project | [record-so101.py](../code/project/record-so101.py) |
| project | [release_offline_check.py](../code/project/release_offline_check.py) |
| project | [requirements-windows.lock.txt](../code/project/requirements-windows.lock.txt) |
| project | [run.ps1](../code/project/run.ps1) |
| project | [so101_paths.py](../code/project/so101_paths.py) |
| project | [test-yellow-smolvla-robot.py](../code/project/test-yellow-smolvla-robot.py) |
| project | [train-yellow-smolvla.py](../code/project/train-yellow-smolvla.py) |
| project | [training-reference.json](../code/project/training-reference.json) |
| project | [yellow-model-start-pose.json](../code/project/yellow-model-start-pose.json) |
| project | [yellow_action_queue.py](../code/project/yellow_action_queue.py) |
| project | [yellow_start_pose.py](../code/project/yellow_start_pose.py) |
| lerobot | [camera_opencv.py](../code/lerobot/camera_opencv.py) |
| lerobot | [configuration_smolvla.py](../code/lerobot/configuration_smolvla.py) |
| lerobot | [dataset_reader.py](../code/lerobot/dataset_reader.py) |
| lerobot | [flow_matching.py](../code/lerobot/flow_matching.py) |
| lerobot | [lerobot_dataset.py](../code/lerobot/lerobot_dataset.py) |
| lerobot | [lerobot_teleoperate.py](../code/lerobot/lerobot_teleoperate.py) |
| lerobot | [lerobot_train.py](../code/lerobot/lerobot_train.py) |
| lerobot | [modeling_smolvla.py](../code/lerobot/modeling_smolvla.py) |
| lerobot | [processor_smolvla.py](../code/lerobot/processor_smolvla.py) |
| lerobot | [smolvlm_with_expert.py](../code/lerobot/smolvlm_with_expert.py) |
| lerobot | [utils.py](../code/lerobot/utils.py) |
| lerobot | [vla_utils.py](../code/lerobot/vla_utils.py) |

这些是与教程对应的阅读快照，GitHub代码页可直接按`#L行号`定位；它们不是独立安装包。项目实际运行与硬件配置仍按仓库根README和已校准设备设置。官方快照保留原许可证与来源。

## 现有文档怎样组合

| 文件 | 保留角色 |
|---|---|
| `机械臂项目记录.md` | 当前有效配置与结论的主记录；后续进度在这里更新 |
| `SmolVLA从0到1实施教程.md` | 本文：机制、数据、代码、流程和问题修复的完整讲解 |
| `实施教程/index.html` | 同一正文的离线图解阅读版 |
| `数据验收报告.md` | 50条逐条备注和验收证据 |
| `黄色螺母训练流程.md` | 训练阶段的操作、路径和日志索引 |
| `第5000步模型离线检查.md` | 部署前离线预测、误差及边界检查的阶段报告 |
| `黄色螺母模型实机测试说明.md` | 测试按键、保护和执行器代码索引 |
| `发布验收/` | 平台上传校验与展示证据 |

阶段报告中的“下一步”“尚未启动”描述其编写时点；当前状态以主记录和本文开篇为准。尤其不能继续把离线检查的早期短窗口建议当成v7当前配置。原文档未被删改，本章阶段报告列表提供统一阅读入口。

- [机械臂项目记录.md](../references/机械臂项目记录.md)
- [数据验收报告.md](../references/数据验收报告.md)
- [黄色螺母训练流程.md](../references/黄色螺母训练流程.md)
- [第5000步模型离线检查.md](../references/第5000步模型离线检查.md)
- [黄色螺母模型实机测试说明.md](../references/黄色螺母模型实机测试说明.md)

阶段报告中的“下一步”“尚未启动”描述编写时点，当前状态以教程开篇和项目主记录为准；不能把早期短窗口建议当作v7当前执行配置。原件、各阶段证据与本轮发布校验随离线档案保存。

## 官方与本项目的边界

官方提供预训练权重、策略结构、数据读取、处理器、Flow Matching、优化器/调度器及训练/检查点基础；本项目负责硬件坐标映射、可靠连续采集、明确划分和统计、训练可观察性、受限异步执行、夹爪解耦、固定起点和复现包装。教学动画只解释这些机制，不冒充官方示意图或实测轨迹。

官方深入阅读：[SmolVLA版本文档](https://huggingface.co/docs/lerobot/v0.6.1/en/smolvla)、[SmolVLA论文](https://arxiv.org/abs/2506.01844)、[v0.6.1策略源码](https://github.com/huggingface/lerobot/tree/v0.6.1/src/lerobot/policies/smolvla)、[LeRobot v3格式](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)。教程中的参数、损失和时序细节优先依据保存配置与本次读取的对应版本源码，而不是将最新版文档行为自动套用到旧环境。

离线交互包无需CDN、账号或联网，所有图片、视频、正文、指标和代码附件在本地。原HEIC/MOV保留下载目录；展示使用方向正确、去EXIF的JPG及完整55.33秒H.264副本，未裁剪动作顺序。

---

**停一下，用自己的话回答：** 阅读代码快照能否代替部署包和已验证硬件配置？

<details>
<summary>参考答案</summary>

不能。快照用于理解与定位；运行还需要匹配依赖、完整资产及自己的现场配置。

</details>

[教程目录](../README.md) · [上一章](19-reproduce.md)
