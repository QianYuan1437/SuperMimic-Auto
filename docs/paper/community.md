# 社区复现与扩展

> 数据采集时间：2026-11-18。star 数来自 GitHub Search API（`q=supermimic&sort=stars`，共 206 个结果）；描述和数字来自各仓库 README（`raw.githubusercontent.com`）或 Release 页面。
> **除官方仓库外，所有性能数字都是各仓库自己报告的（README 自述），本文没有独立验证。** BONES-SEED 基准评测的是训练动作，衡量的是覆盖度，不是泛化能力（论文 App. F.2，awesome-supermimic 也特别提醒了这一点）。不同仓库的过滤后片段数（129,514 / 129,491 / 129,487）、评测协议和硬件都不同，数字不能直接横向比较。

## 1. 官方资源

| 资源 | 内容 | 来源 |
|---|---|---|
| [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)（★1785，Apache-2.0，2026-11-10 创建） | 官方训练代码，包含 `SuperMimic-GenCoord-G1` 和 `SuperMimic-MaxCoord-G1` 两个任务，以及 `train.py`、`play.py --wandb-path`。**没有 Release，没有发布 checkpoint，没有部署代码**，README 也没有讲数据怎么获取 | README；`/releases` 页面显示 “There aren't any releases here” |
| [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research)（★47） | “mjlab with the motion-tracking extensions SuperMimic uses”，需要和 supermimic 并列克隆到同一目录下 | 仓库描述；supermimic README |
| [supermimic.github.io](https://supermimic.github.io/) | 论文摘要、FAQ、视频、浏览器 Web demo | 项目页 |
| Web demo 权重 | **不开放**。[demo/LICENSE](https://supermimic.github.io/demo/LICENSE.html) 写明 “All rights reserved”，对 policy 和 world-model 参数“no licence is granted” | demo LICENSE 页 |

## 2. 关键资源获取指南

### 2.1 BONES-SEED 数据集

| 项 | 内容 | 来源 |
|---|---|---|
| 地址 | <https://huggingface.co/datasets/bones-studio/seed> | 论文参考文献 [28]；HF |
| 访问方式 | **Gated**（HF API 返回 `gated: auto`）。申请时要填写姓名、单位、邮箱，并确认自己是学术用户，或所在公司年收入低于 100 万美元。许可协议：<https://bones.studio/info/seed-license> | HF 页面；社区调研 |
| 规模 | 142,220 条动作（71,132 条原始 + 71,088 条镜像），120 fps，约 288 h，522 名演员 | HF 数据卡 |
| G1 重定向文件 | `g1.tar.gz`（23,499,973,647 B，约 23.5 GB），里面是关节角 CSV，路径形如 `g1/csv/{date}/{motion_name}.csv`，由 NVIDIA 提供。训练 G1 只需要这一个文件，`soma_*.tar.gz`（约 45 GB × 2）用不到 | HF 文件树 API |
| 论文的过滤结果 | 142,220 → 129,514 条（264.56 h）。论文给出了过滤规则，**但没有公开脚本和片段列表** | 论文 App. E |
| 社区的现成处理流程 | CuboiLeo：`scripts/data/csv_to_npz_batch.py`、`filter_bones_seed.py`、`build_library.py`。Motphys：`unisupermimic-boneseed all`，一键下载、过滤并打包成 fp16 库，129,487 条，47.3M 帧，6.72 GB。realmimic 和 blinkmimic2 也有各自的脚本 | 各 README |
| 官方代码的接入方式 | `mjlab_research/src/mjlab/scripts/csv_to_npz.py`（默认 30→50 fps，结果上传到 W&B registry），也可以用 `train.py --motion_file a.npz,b.npz` 直接加载本地 npz。注意 BONES-SEED 原始数据是 120 fps，转换时要设 `--input_fps 120` | 官方代码 |
| 资源需求 | blinkmimic2 报告转换耗时约 7 min（12 workers），打包时需要约 13 GB 内存。realmimic 建议预留约 170 GB 磁盘。fp16 动作库常驻 GPU 时约占 6.7 GB 显存 | 各 README |
| 替代的小数据集 | LAFAN1 G1 重定向数据（40 段，2.45 h），可从 <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset> 获取，适合先跑通流程。论文中 1 张 H100 训练 1.2 h 达到回报 0.82 的结果就是在 LAFAN1 上得到的 | awesome-supermimic；论文 §4.1 |

### 2.2 预训练 checkpoint

**官方没有发布 checkpoint。** 下表列出社区发布的权重。

| 仓库 | 文件 | 训练数据与结果（README 自述） | 格式与许可 |
|---|---|---|---|
| [kevinhenryzz/realmimic](https://github.com/kevinhenryzz/realmimic) | HF [kevinhenryz/realmimic](https://huggingface.co/kevinhenryz/realmimic)：`realmimic_g1_86k.pt`、`web/policy.onnx`、`web/world_model.onnx` | BONES-SEED 全集，**1× RTX 4090 训练 10.5 h**，86k 迭代，回报 0.9545，129,491 条中失败 10 条 | `.pt` + ONNX；CC BY-NC 4.0 |
| [NJU-…/SuperMimic-Repro-Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/releases) | v1.0：`e11c_0220000.tar.gz`、`e13i_0291000.tar.gz`；v1.1：`e24_0355000_lean35kfs.tar.gz` | BONES-SEED，回报 0.93，约 13 万条中失败 46 条。e13i 额外加了 4 段舞蹈，e24 是转体舞蹈专用 | 每个包里都有训练存档和可部署 ONNX（附带 `policy_ir9.onnx`） |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced/releases/tag/e8-boneseed-100k) | `e8_boneseed_ckpt_0100000.pt` + `config.json` | BONES-SEED，4× RTX 5090 训练 7.1 h，100k 迭代，回报 0.930，129,491 条中失败 145 条。awesome-supermimic 指出：0.930 是在 129,943 条上测得的，在过滤后的集合上是 0.932 | `.pt`（只适配该仓库自己的代码） |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic/releases/tag/checkpoints-v1) | `boneseed_ckpt_012000.pt`、`boneseed_ema_nosync_ckpt_008000.pt`、`lafan1_ckpt_012000.pt` | BONES-SEED 12k 迭代：回报 0.9243，失败 350/129,487；EMA 8k 迭代：回报 0.9395，失败 193 条。README 写的硬件是 “RTX 4090 (48 GB)”，标准 4090 只有 24 GB | `.pt`（UniLab） |
| [jyl681/blinkmimic2](https://github.com/jyl681/blinkmimic2) | Release v1.0.0：`blinkmimic2_seed{1,2,3}.pt` | BONES-SEED 全集，1× RTX 4090 训练 15.5 min，3 个种子平均失败 17/129,491，回报 0.946。方法上做了大量加速修改 | `.pt`；CC BY-NC 4.0 |
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | 仓库内 `checkpoints/lafan1_g1_it10000.pt` | 仅 LAFAN1：RTX PRO 6000 训练 0.95 h，回报 0.842，失败 0/40 | `.pt`（Isaac Lab 3） |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab) | 仓库内 `checkpoints/best/policy_best.pt`、`world_model_best.pt` | 仅 LAFAN1：回报 0.8528，40/40 全部成功 | `.pt` |
| [KyrieYoung926/supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy) | Google Drive 上的 [`policy.onnx`](https://drive.google.com/drive/folders/1c8pdlvFXPxH9Bys2G5XvkD0AF7bZ8QCy)（180000 步导出版本） | 基于 CuboiLeo 代码训练 | ONNX，只能用于部署 |

> 社区 checkpoint 的观测布局、动作缩放和 PD 增益都和各自的代码绑定，**不能直接加载到官方 `supermimic` 代码里**。

### 2.3 Sim2Real / 部署代码

官方仓库没有部署代码，论文也没有描述实机软件栈。

| 仓库 | 机器人 | 部署链路 | 要点（README 原文） |
|---|---|---|---|
| [NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real) | **G1 29-DoF 实机** | 导出 ONNX → `scripts/deploy/sim2sim.py`（加延迟、负重、摩擦扰动做验收）→ `scripts/deploy/bridge/g1_runner_bridge.py`（基于 unitree_sdk2） | 训练端要求 NVIDIA 驱动 ≥580；机器人端只需要 numpy、onnxruntime 和 Unitree SDK。PC2 上的 onnxruntime 版本较旧，要把 `policy_ir9.onnx` 复制成 `policy.onnx`。必须有吊架和遥控器，按 L2+B 进入阻尼（软急停）。倾斜超过 20° 时自动停止。文档：`docs/SIM2SIM.md`、`docs/DEPLOY_G1.md`。实机跳完了一段约 46 s 的舞蹈 |
| [KyrieYoung926/supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy) | G1 29-DoF（MuJoCo 和实机） | `unitree_sdk2_python`（固定在 commit `814556d`）+ CycloneDDS + ONNX | 结构轻量，策略 ONNX 从 Google Drive 下载 |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab) → [RoboJuDo-Plus `supermimic` 分支](https://github.com/chifongip/RoboJuDo-Plus/tree/supermimic) | **G1 23-DoF**（不是 29-DoF） | RoboJuDo-Plus 框架，支持 loco/mimic 切换 | 手柄 Y 回到默认姿态，B 进入阻尼，A 进入零力矩。作者说明速度跟踪功能还在开发中 |
| [kevinhenryzz/realmimic](https://github.com/kevinhenryzz/realmimic) | G1（目前只在仿真中验证） | 通过 NVIDIA SONIC runtime 运行 | 适配真机的权重计划在 v1.1 发布。README 提到训练用的是 mjlab 默认低增益（髋 pitch kp≈40），SONIC runtime 用的是 kp 99 |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | — | `sim2sim.py`，只做策略输入输出、网络形状和资产一致性检查 | 不包含实机部署 |
| 官方代码自带 | — | `runners/exporter.py:export_actor_as_onnx`（opset 18，附带 mjlab tracking 元数据） | 只在 `actor_obs="proprio"` 且动作帧数 ≤15,000 时自动导出，详见 [论文精读](./notes.md) §2.9 |

## 3. 主要社区仓库一览

按 star 数排序，数据截至 2026-11-18。

| 仓库 | ★ | 框架 / 方向 | 一句话说明（依据 README） | 数据 / 权重 / 部署 |
|---|---|---|---|---|
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | 907 | Isaac Lab 3（kit-less）+ MuJoCo-Warp，Docker | “vibe coded supermimic within 1h”。只复现了 LAFAN1，RTX PRO 6000 上 0.95 h 达到回报 0.842、失败 0/40。需要约 20 GB 磁盘；Isaac Lab 会预留约 430 GB 虚拟内存，所以不要设置 `ulimit -v` | LAFAN1 / 有权重 / 无部署 |
| [miao-org/awesome-supermimic](https://github.com/miao-org/awesome-supermimic) | 632 | 资源列表（CC0） | 收录了 46 个项目，覆盖实现、扩展、实机和数据，每条都附有 “Upstream report / Useful context” 的事实核查说明 | 索引 |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | 468 | mjlab / MuJoCo Warp | 官方代码发布前完全依据论文写的复现，现已停止维护，作者建议改用官方代码。实现了数据并行、App. F 基准和 BONES-SEED 处理。4×5090 训练 7.1 h，回报 0.930。README 列出的未完成项：速度接口、PPO 基线、ONNX 导出、实机部署 | BONES-SEED / 有权重 / 无部署 |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | 312 | UniLab（mjwarp / MuJoCo / Motrix 后端） | 独立复现，带一键处理 BONES-SEED 的流程和全库评测器（评测时可用 `--num_envs N` 适配小显存）。试验过 EMA、no-sync、Muon 等改进 | BONES-SEED / 有权重 / 只有 sim2sim 检查 |
| [fuyichen0725/blinkmimic](https://github.com/fuyichen0725/blinkmimic) | 298 | 加速变体 | LAFAN1：1× RTX 4090 上 3.4 min 达到回报 ≥0.82 且失败 0/40（3 个种子） | LAFAN1 |
| [Huy-Quang-Dao/instantTrack](https://github.com/Huy-Quang-Dao/instantTrack) | 215 | 加速变体 | LAFAN1：1× RTX 5090 上约 3 min 达到回报 0.82、失败 0/40（5 个种子）。`tools/` 下有 ONNX 导出和检查脚本 | LAFAN1 / 无部署 |
| [CumulusAlpha/Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) | 149 | 扩展：灵巧手 | 把 SuperMimic 范式用到灵巧手操作任务上 | — |
| [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) | 134 | 扩展：四足 Go2 | README 和描述都很少 | — |
| [emoPointer/supermimic_manipulation](https://github.com/emoPointer/supermimic_manipulation) | 112 | 扩展：操作 | 在 mjlab 中用 SuperMimic 世界模型做基于状态的机械臂操作 | — |
| [NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real) | 104 | 基于 CuboiLeo 代码 + 实机 | **完整的 G1 29-DoF sim2real 链路**，中文文档，实机跳舞视频在 Release 中 | BONES-SEED / 有权重和 ONNX / **有实机部署** |
| [Shimingyi/allmimic](https://github.com/Shimingyi/allmimic) | 49 | 重新实现 | 自称 “SuperMimic re-implementation”，推荐的 preset 改动了方法和数据采样 | — |
| [Renforce-Dynamics/trackerlab-supermimic](https://github.com/Renforce-Dynamics/trackerlab-supermimic) | 24 | IsaacLab 2.3.2 / Isaac Sim 5.1 / PhysX，可选 Warp 训练器 | 支持 G1、Go2、A3、T1、Bumi、Pi+ 多种机器人，主要用 LAFAN1。README 明确说不声称复现了论文的 BONES-SEED 基准和实机结果。在 96 GB 卡上跑 8,096 个环境时，PyTorch 峰值显存为 18.72 GiB | 多机器人 / 有 `export_policy.py` |
| [kevinhenryzz/realmimic](https://github.com/kevinhenryzz/realmimic) | 19 | 基于官方思路 | **1× RTX 4090（24 GB）训练 BONES-SEED 全集**，10.5 h，回报 0.9545。附有详细的“与论文差异”表 | BONES-SEED / HF 权重 / 只在仿真中部署 |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab) | 16 | mjlab | LAFAN1 复现加完整的消融链：观测归一化 → 朝向锚点 → WM 归一化 → 修正余弦 LR，回报从 0.60 提升到 0.85。RTX A6000 48 GB 训练 15k 迭代约 4 h。提供一个 “reduced-workload” 配置（64 envs、WM batch 256），但效果未验证 | LAFAN1 / 有权重 / G1 23-DoF 实机 |
| [KyrieYoung926/supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy) | 14 | 部署 | 在 MuJoCo 或 29-DoF G1 实机上运行预训练策略 | 只有 ONNX / 有实机部署 |
| [jyl681/blinkmimic2](https://github.com/jyl681/blinkmimic2) | — | 加速变体 | BONES-SEED 全集 1× RTX 4090 训练 15.5 min（3 个种子平均失败 17 条）。README 写训练需要约 19 GB 显存 | BONES-SEED / 有权重 |
| [kangzejian1896/supermimic-g1-reproduction](https://github.com/kangzejian1896/supermimic-g1-reproduction) | — | 复现 | 1× A800 80GB、4096 个环境，观测到的显存占用约 24 GB。168k 迭代、25.86 h 的中间 checkpoint，使用 84,287 条片段（自己划分的训练集） | Release `v0.2.0-168k` |
| [lucidrains/supermimic-pytorch](https://github.com/lucidrains/supermimic-pytorch) | 11 | 纯 PyTorch | 非官方实现 | — |

上表中不在 search 前 50 结果里的仓库，star 数标为 “—”（API 配额用尽，没有查到）。其他可以关注的方向：苹果芯片（`yusenthebot/macmimic`、`WXyuany/supermimic-m5`）、其他人形机器人（`liuskywalkerjskd/supermimic_oli` LimX Oli、`JzKosann/supermimic-isaaclab-h1_2` H1-2）、感知与地形（`mode1223345/SuperMimic-perceptive`、`yhx1203/supermimic_parkour`）、肌肉驱动（`kaibohex/muscle-supermimic`）。

## 4. 对单卡 24 GB（RTX 3090/4090）复现有用的社区经验

| 经验 | 来源 |
|---|---|
| 策略 batch 从 4096 降到 **2048**，才能和常驻显存的动作库一起放进 24 GB。BONES-SEED 全集在 4090 上训练 10.5 h（86k 迭代，前 69k 迭代内做 LR 衰减） | realmimic README “Differences from the paper” |
| CuboiLeo 训练 BONES-SEED 时也用策略 batch 2048 | CuboiLeo README / 社区调研 |
| 报告的显存占用：约 19 GB（blinkmimic2，经过加速改造）；4096 个环境约 24 GB（kangzejian，A800 上观测）；8,096 个环境 18.72 GiB（trackerlab，PhysX） | 各 README |
| BONES-SEED 的 fp16 动作库约 6.7 GB，常驻 GPU | CuboiLeo、Motphys README |
| 评测时可以减少并行环境数（如 `--num_envs N`），评测在更小的显卡上也能跑 | Motphys、realmimic README |
| 先用 LAFAN1 跑通流程：论文中 H100 上 1.2 h 达到回报 0.82。社区在 RTX 5080 16 GB 上用 2048 个环境跑过 LAFAN1（`rzc156/supermimic-rtx5080`） | 论文 §4.1；社区调研 |

> 3090（Ampere，24 GB，支持 TF32）**没有任何官方或社区的实测数据**。上面 4090 的数字只能作为显存规模的参考。3090 的算力和带宽都低于 4090，训练时间预计会更长，具体要实测。
