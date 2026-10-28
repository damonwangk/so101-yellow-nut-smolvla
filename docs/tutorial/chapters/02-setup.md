<a name="setup"></a>

# 02｜硬件、环境与坐标，先建立共同语言

[教程目录](../README.md) · [上一章](01-roadmap.md) · [下一章](03-capture.md)

> **学习目标：对齐单位与坐标**
>
> 能区分编码器计数、关节角、夹爪开度和主从映射。

## 环境安装解决什么

我们采用 Windows 原生 Python/CUDA 流程，不依赖 ROS 或 WSL。项目内虚拟环境隔离依赖，环境脚本统一配置模型缓存、Torch缓存、FFmpeg及校准路径。这样采集、训练与推理使用同一份依赖和同一份数据契约。

| 组成 | 已使用的版本 / 设置 |
|---|---|
| Python / LeRobot | 3.12.15 / 0.6.1 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| transformers / accelerate | 5.5.4 / 1.15.0 |
| 视频读取 | PyAV 15.1.0；数据训练使用 pyav 后端 |
| 视频编码 | FFmpeg 8.1.2，H.264 |
| 硬件 | SO-101 Leader + Follower；单腕部UVC相机 |
| 相机输入 | RGB 640×480，标称30FPS，随从臂腕部移动 |
| 控制循环 | 目标60Hz；正式采集实测约59.75–59.95Hz |

参考环境先准备Python 3.12和支持该CUDA版本的NVIDIA驱动，再创建项目虚拟环境。公开便携包的安装入口执行下面的核心命令，依赖文件固定完整版本；安装不是每次控制启动都重复执行的步骤。

```powershell
Set-Location D:\so-101
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock.txt
. .\scripts\environment.ps1
```

FFmpeg可执行文件和共享DLL目录加入项目环境；官方模型与tokenizer配置下载至项目models目录后，训练/推理使用本地路径。这里展示已采用流程，本文没有在Mac或Windows重新安装。原机已完成校准，日常启动直接复用；新机器的电机ID、方向、行程和编码参考需要按[官方SO-101设置](https://huggingface.co/docs/lerobot/so101)在现场完成，不把作者补偿数值直接写进去。

参考机主臂当前COM4、从臂COM3；应按USB序列号识别，COM编号和相机索引可能随重插变化。另一台机械臂需要自己的校准和工作范围，不能把作者校准当成通用参数。

## 一个机器人有几套“位置”

电机编码器原始计数经过校准得到关节位置。主臂与从臂的机械结构和编码参考不完全相同，因此遥操作时还要做坐标映射。这里的六维顺序固定为：

```text
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

前五维单位为度；第六维是校准后的夹爪开度百分比点，不是角度，也不是牛顿。跟随配置版本4、坐标版本3；主臂肘读数加 `-90.021978°`，腕屈伸加 `-30.197802°`。肘偏移主要抵消编码参考调整，腕偏移是人工对齐得到的近似补偿，不属于高精度外参标定。

```python
# follow-practical.py / WristMapping.nominal 的核心逻辑
out = {k: float(v) for k, v in action.items()}
for k, delta in self.cfg['offset_degrees'].items():
    out[k] += delta
```

采集中的 `action` 已经是映射、边界和限幅之后的从臂目标。模型学的是从臂坐标，部署时 `ModelMapping.nominal()` 直接返回六维目标。若再次加主臂偏移，动作会系统性偏离；这条坐标契约贯穿全流程。

## 两种控制设置不能混用

采集跟随基线保留身体5°、夹爪2个百分点的相对反馈目标限幅。模型测试按后续明确要求修改：身体继续保留边界、5°相对反馈限幅和15°/秒速度；夹爪只保留校准活动范围。后者不反向修改采集器、纯跟随或校准文件。


## 用一次坐标映射检验接口

假设主臂肘读数为90°（教学假设），当前补偿是−90.021978°，则从臂名义目标约−0.021978°；后续还要经过绝对范围及相对反馈限幅。训练存的是最终发送值，不是90这个主臂原数。换另一台机械臂时，偏移不是通用常量，必须重新确认自己的编码坐标与实际姿态对应关系。这里的“坐标正确”先于“模型是否聪明”。

---

**停一下，用自己的话回答：** 为什么模型目标不能再加主臂腕部偏移？

<details>
<summary>参考答案</summary>

监督action已经是从臂坐标；再加一次会重复补偿并系统性偏离。

</details>

[教程目录](../README.md) · [上一章](01-roadmap.md) · [下一章](03-capture.md)
