# 其他设备如何复现

本发布验证的机械结构是SO-101主从臂，单腕部RGB相机，Windows原生LeRobot0.6.1。机器人基座位置、相机安装、夹爪外形和物体/托盘布局都会影响模型；发布的权重不保证直接适用任意机器人。

## 1. 区分参考资料与自己的配置

`calibration/`和`configs/practical-follow.json`是作者设备的参考标定与映射。文件名`win_leader`/`win_follower`是本项目使用的LeRobot设备ID，不是新设备已经校准的证明。不要把作者的原始编码器偏移直接写到别的机械臂。

现场控制前，先完成自己设备的官方校准和可用主从映射，保存两份校准JSON、USB序列号和端口、从臂可用原始编码范围、角度补偿以及校准文件SHA-256，形成一个已验证的SO-101工作目录。

官方参考：[SO-101](https://huggingface.co/docs/lerobot/so101)。本仓库不提供一键重刷作者电机参考偏移的操作。

## 2. 导入已有且已验证的配置

```powershell
.venv\Scripts\python.exe scripts\configure_hardware.py --from-existing D:\your-verified-so101
```

源目录需要：

- `configs/practical-follow.json`（schema可参考仓库中的配置，端口/序列号/原始范围/补偿必须属于自己的设备）
- `calibration/leader/win_leader.json`
- `calibration/follower/win_follower.json`

导入程序检查源配置记录的校准哈希，复制到发布目录并写本地确认标记；不连接或配置电机。修改配置后需重新审核并导入，不把确认标记上传到公共仓库。

`configs/yellow-model-start-pose.json`中的目标采用训练时的从臂坐标。导入程序不会为新设备自动求解物理坐标映射，也不会把越界目标强行裁剪成另一个起始姿态。若自己的校准坐标或工作边界不兼容，应先完成映射/姿态验证或重新采集，不能直接运行旧目标。

## 3. 首次实机检查

先运行OfflineCheck。确认基座和相机安装、视角、运动路径和所有保护配置后，再打开TestYellowNut。第一次Enter只负责受原Guard保护归位；AT_START后放物，第二次Enter才开始模型。

关节边界和单步限幅不提供环境碰撞检测。回位时保持运动区域无障碍，操作员可随时Q保持；退出释放力矩前托稳从臂。

## 4. 关于夹持力

控制器发出的是位置目标。相同开度在不同接触角度/深度下不保证相同夹持力；当前程序没有力传感器或抓取成功识别。夹爪软件目标仍只限活动范围，官方驱动的电流/力矩保护保持原样。不要把视觉上的滑脱直接等同于电机最大力矩不足。
