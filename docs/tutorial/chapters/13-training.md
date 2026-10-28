<a name="training"></a>

# 13｜8GB训练做了哪些处理

[教程目录](../README.md) · [上一章](12-flow.md) · [下一章](14-evaluation.md)

> **学习目标：理解参数更新与显存取舍**
>
> 知道冻结、反向传播、优化器、学习率和日志各负责什么。

## 先证明能跑，再正式微调

早期用已有试采做100次更新、batch1可运行性验证，检查有限损失/梯度、参数变化和保存读取。它只是确认环境和显存可用，没有把该短跑模型当正式模型。正式任务使用新的黄色螺母50条数据，选batch8。

## 训练范围与显存

`train_expert_only=True` 冻结整个VLM，`freeze_vision_encoder=True` 保持视觉编码器冻结与eval状态；状态投影、动作输入/输出投影、时间MLP和动作专家参与更新。冻结节省这些参数的梯度与优化器状态，但前向计算仍然存在，训练状态投影的梯度也需要经过相应计算图。

采用CUDA BF16 autocast、无梯度累积、`num_workers=0` 与已验证的PyAV解码方式。没有采用LoRA、4bit量化、DeepSpeed、多GPU或CPU卸载；不要把本机能跑归因于没有使用的技术。正式训练PyTorch分配显存峰值约2.56GiB，不含驱动、桌面与所有进程的总显存。

| 训练项 | 最终采用 |
|---|---|
| 初始化 | 本地官方smolvla_base完整权重 |
| Batch / 随机种子 | 8 / 1000 |
| 优化器 | AdamW；β=(0.9,0.95)，eps=1e-8，weight_decay=1e-10 |
| 梯度裁剪 | 全局范数上限10 |
| 学习率 | 峰值1e-4，1000步预热，原20000步余弦计划衰减至2.5e-6 |
| 原预算 / 实际停止 | 20000 / 12000步；停止时没有重算余弦计划 |
| 验证 / 保存 | 每1000步验证完整5条；100步首存，此后每1000步保存 |
| 外部上传 | 训练期间Hub与WandB关闭，使用本地权重和数据 |

## 一步更新的调用顺序

```text
取batch → 图像uint8/255 → preprocessor
→ policy.forward（随机t和噪声、有效掩码MSE）
→ accelerator.backward(loss)
→ clip_grad_norm_(…,10)
→ optimizer.step() → optimizer.zero_grad()
→ scheduler.step() → 记录loss/grad_norm/lr/显存/耗时
```

`train-yellow-smolvla.py` 是项目封装，实际梯度更新、验证和检查点格式复用官方 `lerobot_train.py`。封装接入确定的episode名单、训练集统计、本地tokenizer、日志和保存核验；没有重写神经网络或另造训练框架。

12000步指12000次优化更新，不是12000条示范或12000个episode。按batch8计算约消费96000个锚点样本，约为26722训练帧的3.59倍；采样次序、末批等会影响精确对应，重叠的50步窗口也不能当成相互独立的新任务。每次重新抽取噪声和t，所以同一观测再次出现时，Flow Matching训练输入仍会变化。

## 日志、续训与完整检查点

当前封装的 `monitored_update()` 在前5步及之后每25步直接 `print(..., flush=True)`，输出loss、梯度范数、学习率、GPU峰值及单步时间。`ConsoleTee` 同时保留控制台文件；`metrics.jsonl`逐事件持久化，状态文件用临时文件替换。此直接打印改动供后续运行使用；本轮实际曲线来自已保存metrics，而不是重新训练得到。

完整检查点包含模型、预/后处理器、归一化张量、训练配置，以及优化器、调度器、步数与随机状态。Windows使用 `last_checkpoint.json` 指针避免依赖符号链接权限；防重复进程锁避免两个训练相互写目录。用户要求在12000步结束时，先确认验证与完整保存完成，再终止进程；`early_stop_result.json`阻止误续训该已结束轮次。

---

**停一下，用自己的话回答：** 冻结VLM之后，为什么仍需计算VLM前向？

<details>
<summary>参考答案</summary>

动作专家仍需它的条件特征；冻结只停止相应参数更新，不会删掉视觉/语言计算。

</details>

[教程目录](../README.md) · [上一章](12-flow.md) · [下一章](14-evaluation.md)
