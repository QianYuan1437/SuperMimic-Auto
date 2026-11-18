# SuperMimic Auto

**SuperMimic（CoRL 2026 Oral）一键部署与复刻指南**：用一个可微世界模型训练 Unitree G1 人形机器人全身动作跟踪。任意一台带 NVIDIA GPU 的 Linux 电脑，按本仓库的脚本和文档即可从零复现。

📖 **文档站点：<https://qianyuan1437.github.io/SuperMimic-Auto/>**（与本仓库 `docs/` 内容相同）

| | |
| --- | --- |
| 论文 | [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480)（Liao, Shao, Long, Zakka, Horowitz, Abbeel, Sreenath · UC Berkeley） |
| 项目主页 | <https://supermimic.github.io/> |
| 官方代码 | [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) @ `7e94b07` + [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research) @ `6ea3e57` |

## 快速开始

```bash
git clone https://github.com/QianYuan1437/SuperMimic-Auto.git && cd SuperMimic-Auto
bash scripts/00_check_env.sh          # 检查 GPU / 驱动(≥580，推荐≥595) / uv / 磁盘
bash scripts/01_setup.sh              # 克隆固定版本官方代码 + uv sync（国内加 UV_MIRROR=...）
bash scripts/02_download_data.sh      # LAFAN1 → G1 的 40 段动作（国内加 HF_ENDPOINT=https://hf-mirror.com）
bash scripts/03_convert_motions.sh    # CSV(30fps) → NPZ(50fps)
PRESET=single ITERS=20000 bash scripts/04_train.sh   # 单段动作快速验证
PRESET=lafan bash scripts/04_train.sh                # 全部 40 段（2.45 小时动作）
bash scripts/06_eval.sh               # 批量评估：成功率 / MPKPE → eval.json
bash scripts/07_record.sh "" --clip dance1_subject1  # 离屏录像
bash scripts/05_play.sh               # 交互查看器 http://localhost:8080
```

不需要 W&B 账号：官方代码从作者私有的 W&B registry 拉取动作、把权重上传 W&B，本仓库的 `tools/` 把这些全部换成了本地文件。

## 仓库结构

```text
configs/upstream.env   上游仓库与固定提交号（唯一版本来源）
scripts/               00~08 按步骤编号的 Shell 入口
tools/                 本地化训练 / 回放 / 评估 / 录像 / 画图（包装上游，不改上游代码）
docker/                Dockerfile 与 GPU 容器启动脚本
docs/                  分步文档（也是 GitHub Pages 站点源文件）
  guide/               1 硬件要求 → 2 安装 → 3 数据 → 4 训练 → 5 评估 → 6 部署 → 7 Docker → 8 排错
  paper/               论文精读与代码对照、社区复现与扩展
  results/             本机 RTX 3090 复现记录
  reference/           脚本一览、超参数、仓库结构
results/               本机复现产物（评估 JSON、曲线）
```

## 文档目录

- 部署指南：[快速开始](docs/guide/quickstart.md) · [硬件与系统要求](docs/guide/requirements.md) · [安装](docs/guide/install.md) · [数据](docs/guide/data.md) · [训练](docs/guide/train.md) · [回放评估录像](docs/guide/evaluate.md) · [导出与真机部署](docs/guide/deploy.md) · [Docker](docs/guide/docker.md) · [常见问题](docs/guide/troubleshooting.md)
- 论文：[论文精读与代码对照](docs/paper/notes.md) · [社区复现与扩展](docs/paper/community.md)
- 结果：[RTX 3090 复现记录](docs/results/index.md)
- 参考：[脚本与工具](docs/reference/scripts.md) · [训练超参数](docs/reference/config.md) · [仓库结构](docs/reference/layout.md)

## 许可

本仓库的脚本与文档采用 [MIT](LICENSE)。SuperMimic 与 mjlab 官方代码（由脚本克隆到 `third_party/`，不随本仓库分发）为 Apache-2.0，© 各自作者。LAFAN1 数据集与重定向数据遵循其各自的许可。

如果本仓库对你有帮助，请引用原论文：

```bibtex
@inproceedings{liao2026supermimic,
  title     = {{SuperMimic}: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author    = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
