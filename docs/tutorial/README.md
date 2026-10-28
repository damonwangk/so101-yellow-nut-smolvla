# SO-101 × SmolVLA：从一颗黄色螺母入门具身智能

[返回项目与一键启动](../../README.md) · [下载完整离线交互教程](https://github.com/damonwangk/so101-yellow-nut-smolvla/releases/download/tutorial-v1.0/so101-smolvla-tutorial-offline.zip)

这是一份基于真实SO-101项目的21章中文教程：50条示范、单腕部RGB与6维状态/目标、RTX 5060 Laptop 8GB微调、5000步模型、受限v7执行器，以及操作员确认的黄色螺母抓取落盘案例。

**阅读基线：** LeRobot0.6.1及本项目保存配置。教程内容截至2026-10-22，2026-10-28转换为GitHub Markdown；没有新增训练、模型推理或硬件试验。一次参考成功不等于统计成功率或跨场景泛化。

## 怎么读

- 第一次入门：01学习路线 → 04频率与时间对齐 → 06数据工程 → 08真实batch → 09模型基础 → 10架构 → 12Flow Matching。
- 复现系统：02环境/坐标 → 03采集 → 07划分 → 13训练 → 15部署 → 16执行 → 17固定起点。
- 定位问题：18排坑 → 14证据边界 → 19下一轮实验。

每章保留正文、学习目标、自测及答案。原网页的21个交互区块转换为静态图、真实数值表、代码和Mermaid；所有滑块与动画保留在上方下载包，解压后打开`开始阅读.html`即可，无需安装模型环境。

## 章节目录

| 章 | 内容 |
|---|---|
| 00 | [先看我们实际完成了什么](chapters/00-overview.md) |
| 01 | [具身智能入门：用这个任务建立学习路线](chapters/01-roadmap.md) |
| 02 | [硬件、环境与坐标，先建立共同语言](chapters/02-setup.md) |
| 03 | [采集器怎样把演示变成可训练数据](chapters/03-capture.md) |
| 04 | [频率与时间对齐：沿真实日志走一遍](chapters/04-timing.md) |
| 05 | [LeRobot v3 文件里究竟是什么](chapters/05-format.md) |
| 06 | [数据工程师怎样判断一条示范是否可信](chapters/06-data-engineering.md) |
| 07 | [验收、划分与未来动作窗口](chapters/07-dataset.md) |
| 08 | [动手追踪：你的8张图怎样成为一个真实batch](chapters/08-batch-lab.md) |
| 09 | [读懂模型所需基础：Tensor、MLP与Transformer](chapters/09-foundations.md) |
| 10 | [SmolVLA 的主要架构与输入输出](chapters/10-architecture.md) |
| 11 | [一个batch进入网络后，逐层怎样流动](chapters/11-forward.md) |
| 12 | [训练到底优化什么：Flow Matching](chapters/12-flow.md) |
| 13 | [8GB训练做了哪些处理](chapters/13-training.md) |
| 14 | [为什么选择5000步，离线验收又证明什么](chapters/14-evaluation.md) |
| 15 | [训练完成后，模型怎样落到真机上](chapters/15-deployment.md) |
| 16 | [最关键的一层：让预测轨迹真的被执行](chapters/16-execution.md) |
| 17 | [固定起点与两次Enter，减少起始不确定性](chapters/17-homing.md) |
| 18 | [一路遇到的问题与最终解决方案](chapters/18-pitfalls.md) |
| 19 | [如何复现，以及怎样做下一轮实验](chapters/19-reproduce.md) |
| 20 | [代码阅读地图、文件组合与来源](chapters/20-sources.md) |

## 图像、数据与代码

- [真实采集/推理时间证据](assets/timing-real.json)：29570行聚合、episode22全部490帧时间，以及参考测试的控制/预测记录。
- [真实batch快照](assets/batch-real.json)：8个锚点、50×6动作、mask、真实token及标准化前后值。
- [固定源码阅读快照](code/)：31份项目/官方文件和上游许可证；与仓库实际运行入口区分。
- [阶段报告](references/)及[验收/来源记录](provenance/)；既有原始数据与模型版本保持不变。
- [成功视频](https://github.com/damonwangk/so101-yellow-nut-smolvla/releases/download/v1.0.0/success-demo.mp4)沿用既有Release，不复制大型视频到Git历史。

## 为什么采用Markdown

参考[Microsoft AI for Beginners](https://github.com/microsoft/AI-For-Beginners)的章节README与目录结构，并对照[Hugging Face Course](https://github.com/huggingface/course)的课程内容组织。GitHub原生支持Markdown的代码、表格、图片、折叠内容与[Mermaid图](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams)，因此主阅读采用分章`.md`；需要JavaScript的交互另以下载包保存，不依赖仓库页面执行脚本。

## 维护与来源

直接修改本目录章节Markdown即可；真实样例JSON和源码快照保持独立，并更新来源与文件校验清单。离线ZIP保存转换前完整网页、构建脚本、原始Markdown及附件，用于恢复原交互版本。Markdown发布清单见[publication.json](provenance/publication.json)。

原创讲解/图解及项目代码沿用本仓库许可证；上游源码和依赖保留各自许可，见[代码快照许可证](code/LICENSE)及[仓库LICENSE](../../LICENSE)。所有图形明确区分真实数据、源码结构推导与教学算例。
