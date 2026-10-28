<a name="deployment"></a>

# 15｜训练完成后，模型怎样落到真机上

[教程目录](../README.md) · [上一章](14-evaluation.md) · [下一章](16-execution.md)

> **学习目标：把模型作为完整软件组件部署**
>
> 知道权重、处理器、配置、坐标和日志需要一起匹配。

## 需要带走的不只一个权重文件

选中检查点的 `pretrained_model/` 至少包含 `model.safetensors`、`config.json`、`train_config.json`、`policy_preprocessor.json`、`policy_postprocessor.json`，以及normalizer/unnormalizer的safetensors。再加本地tokenizer/processor配置、项目控制器、已校准硬件与匹配的坐标/边界配置。

权重约1.198GB；只复制它会丢失输入键、50步配置、归一化和tokenizer等部署契约。公开模型仓库已一起保存这些文件。推理不需要训练优化器状态；断点续训需要完整训练检查点。

## 推理进程的核心代码

```python
# test-yellow-smolvla-robot.py / worker.predict，按当前代码摘取
batch = {
    'observation.images.wrist_camera':
        torch.from_numpy(image.copy()).permute(2, 0, 1)
             .float().unsqueeze(0) / 255,
    'observation.state': torch.tensor(state, dtype=torch.float32).unsqueeze(0),
    'task': [TASK],
}
batch = pre(batch)
model.reset()
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    action = model.predict_action_chunk(batch)
action = post(action).float().cpu().numpy()[0]
if action.shape != (50, 6) or not np.isfinite(action).all():
    raise ValueError('Invalid model action chunk')
```

初始化采用 `SmolVLAPolicy.from_pretrained(..., strict=True, local_files_only=True)`；模型eval、BF16、禁用compile，预/后处理器从同一检查点加载。本地做3次黑图热身，随后启动前真实画面影子预测仅验证输入输出，不把该预测作为运动命令。

`model.reset()` 清空的是官方策略内部动作队列；我们每次直接取完整 `predict_action_chunk()`，由外部 `RetimedActionQueue` 管理机器人轨迹。它不会重置模型权重，也不会让机器人自动回零。

## 控制和推理分开运行

Windows用spawn子进程加载模型；父进程以目标60Hz运行软件反馈/目标发送循环，子进程负责GPU推理。请求队列容量1，结果队列容量2，避免堆积大量过期观察。请求带本轮epoch、图像、状态、观察时间和 `request_progress`；返回预测带相同标记与推理耗时。Q、R、新轮次会清空轨迹并改变epoch，旧轮结果不可再次生效。

每次请求还保存真实腕部JPEG，预测事件记录状态、目标、时刻和路径；控制日志逐周期保存 `requested` / `sent` / `observation`、限幅比例与队列进度。诊断可从图像追到命令，不依赖后来手机视频反推控制量。测试日志与训练数据隔离，不自动把测试片段加入训练。

这里的异步执行器是本项目实现，借鉴了“控制与推理解耦”的思路；当前 `rtc_config=null`，没有启用官方RTC去噪指导或照搬官方Policy Server。不能把二者当成同一套实现。


## 一次真正的推理请求与结果，分别装什么

真实日志的推理耗时与持续控制见[第04章](04-timing.md#timing-online)。当前部署取latest图像和current state，配对规则与采集历史回溯不同；`observed_at=min(...)`只是保守年龄检查，不会把两路插值到同一曝光时刻。

```python
# 输入结构示意；图像/状态绑定在同一次请求中，各自采样时刻仍可不同
request = dict(epoch=trial_id, request_progress=policy_phase,
               observed_at=host_time, image=rgb_480x640x3,
               state=six_follower_positions)
# worker计算后：
result = dict(epoch=trial_id, request_progress=policy_phase,
              observed_at=host_time, actions=array_50x6,
              inference_s=elapsed)
```

这份协议把“是哪一轮”“观察多旧”“预测对应哪段策略进度”一并带回。假设Q已把epoch从7改成8，迟到的epoch7结果就应丢弃。否则即使模型本身预测合理，旧轮动作仍可能突然接管新轮机械臂。以上编号为教学例，真实程序每次请求会写入日志，供复查。

---

**停一下，用自己的话回答：** 只复制model.safetensors，为什么不够？

<details>
<summary>参考答案</summary>

会丢失图像键、输入形状、标准化统计、tokenizer和动作窗口等契约；同权重也可能因预处理不同而行为改变。

</details>

[教程目录](../README.md) · [上一章](14-evaluation.md) · [下一章](16-execution.md)
