---
layout: home

hero:
  name: SuperMimic Auto
  text: 一个可微世界模型，训练人形全身动作跟踪
  tagline: SuperMimic（CoRL 2026 Oral）的一键部署与复刻指南：固定版本的官方代码 + 离线化工具链 + 中文分步文档。任意一台带 NVIDIA GPU 的 Linux 电脑，照着做即可在 Unitree G1 上训练出动作跟踪策略。
  actions:
    - theme: brand
      text: 快速开始
      link: /guide/quickstart
    - theme: alt
      text: 本机复现结果
      link: /results/
    - theme: alt
      text: 论文解读
      link: /paper/notes

features:
  - title: 固定版本，可重复
    details: 官方 supermimic 与 mjlab_research 锁定到具体提交，uv.lock 锁定全部 143 个 Python 包，换一台机器得到同样的环境。
    link: /guide/install
  - title: 无需 W&B 账号
    details: 官方代码依赖私有 W&B registry 下载动作、上传权重。tools/ 把数据、权重、配置与曲线全部改为本地文件，完全离线可用。
    link: /reference/scripts
  - title: 开箱即用的数据
    details: 自动下载 LAFAN1 → Unitree G1 的 40 段重定向动作（2.45 小时），用 mjlab 官方算法转换为训练格式。
    link: /guide/data
  - title: 训练 · 评估 · 录像
    details: 按步骤编号的脚本：单动作快速验证、全量训练、按窗口统计成功率与 MPKPE、离屏录制视频、画训练曲线。
    link: /guide/train
  - title: 面向真机
    details: 一个开关训练只用本体感知 + 10 帧历史的可部署策略，自动导出带元数据的 ONNX，并整理了社区 G1 sim2real 方案。
    link: /guide/deploy
  - title: 论文与代码对照
    details: 训练循环、世界模型、BPTT、两种状态表示、基于回报的重置采样……每个概念都对应到具体文件与函数。
    link: /paper/notes
---
