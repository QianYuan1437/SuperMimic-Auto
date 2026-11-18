# RTX 3090 复现记录

本页记录 2026-11-18 在一台消费级工作站上**严格按本仓库脚本**从零复现 SuperMimic 的全过程与结果。所有原始数据（评估 JSON、训练指标、配置）都保存在仓库的 [`results/`](https://github.com/QianYuan1437/SuperMimic-Auto/tree/main/results) 目录。

## 结论速览

| | 结果 |
| --- | --- |
| 数据 | LAFAN1 → G1 全部 40 段动作，2.45 小时，441,140 帧 |
| 训练 | 30,000 次迭代，**3.80 小时**（单张 RTX 3090，GenCoord，4096 环境，官方默认超参数） |
| 0.66 小时 | 902 个 10 秒窗口中 880 个跟踪成功（97.6%） |
| 1.94 小时 | 900/902（99.8%） |
| 3.19 小时 | **902/902（100%）**，40 段动作全部无失败 |
| 最终（3.80 小时） | 4 次重复评估失败窗口数 3 / 1 / 0 / 3（≥ 99.7%），根相对关键点误差 3.1 cm |

<div class="video-grid">
  <DemoVideo src="/media/lafan-dance2_subject1.mp4" caption="dance2_subject1（舞蹈）" />
  <DemoVideo src="/media/lafan-fallAndGetUp1_subject1.mp4" caption="fallAndGetUp1_subject1（倒地起身）" />
  <DemoVideo src="/media/lafan-jumps1_subject1.mp4" caption="jumps1_subject1（跳跃）" />
  <DemoVideo src="/media/lafan-fightAndSports1_subject1.mp4" caption="fightAndSports1_subject1（格斗与运动）" />
  <DemoVideo src="/media/lafan-sprint1_subject2.mp4" caption="sprint1_subject2（冲刺）" />
  <DemoVideo src="/media/lafan-walk3_subject2.mp4" caption="walk3_subject2（行走）" />
</div>

视频由 `scripts/07_record.sh` 离屏录制（3.80 小时的最终策略，每段 12 秒）。实体为物理仿真中由策略控制的 G1，绿色半透明为参考动作；两者之间的偏移主要是全局位置漂移，姿态本身跟得很紧。

## 实验环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 24 GB（同时驱动桌面显示） |
| 驱动 | 595.99.02（CUDA 13.2） |
| CPU / 内存 | Intel Core i7-14700K（28 线程）/ 62 GB |
| 系统 | Ubuntu 24.04.5 LTS |
| Python 环境 | uv 0.12.2，Python 3.13，torch 2.15.0+cu132，mujoco-warp 4.0.1，warp 1.19.0 |
| 上游代码 | supermimic `7e94b07`，mjlab_research `6ea3e57` |

## 部署各步骤耗时

| 步骤 | 耗时 | 备注 |
| --- | --- | --- |
| `00_check_env.sh` | 数秒 | |
| `01_setup.sh` | 约 45 分钟 | 几乎全部是下载约 3 GB 的 torch / CUDA 轮子；直连 PyPI 约 1 MB/s，换华为云镜像后约 3 MB/s。`.venv` 占用 6.4 GB |
| `02_download_data.sh` | 约 2 分钟 | 40 个 CSV，87 MB |
| `03_convert_motions.sh` | 58 秒 | 输出 120 MB NPZ |
| Warp kernel 首次编译 | 约 1 分钟 | 缓存在 `~/.cache/warp`（53 MB），之后启动只需数秒 |

## 实验一：单段动作快速验证

`PRESET=single`（run1_subject5，约 4 分钟的跑步动作），训练 3,000 次迭代（约 25 分钟）后停止。

- 第 970 次迭代左右（约 8 分钟）平均回合长度就达到 480/500 步；
- 评估：**24/24 个窗口全部成功**，根相对关键点误差 4.4 cm，末端位置误差 5.6 cm。

<DemoVideo src="/media/smoke-run1.mp4" caption="单段动作 run1_subject5，训练 3,000 次迭代" />

## 实验二：LAFAN1 全部 40 段动作

### 训练命令

```bash
PRESET=lafan ITERS=30001 SNAPSHOT_EVERY=5000 RUN_NAME=lafan-3090 bash scripts/04_train.sh
bash scripts/09_eval_snapshots.sh   # 训练结束后评估 6 个快照，共 2 分 10 秒
```

除迭代数外全部使用官方默认超参数（GenCoord 表示、`actor_obs=rep_state`、4096 环境、策略 batch 4096、想象视界 32）。

### 速度与资源

| 指标 | 数值 |
| --- | --- |
| 吞吐 | 约 2.2 次迭代/秒（约 9,000 次仿真步/秒） |
| 时间占比 | 仿真采集 6.7% · 世界模型拟合 40.5% · 策略 BPTT 52.7% |
| 显存 | 约 18.9 GB（4096 环境） |
| 总耗时 | 18:17:31 → 22:05:54，3 小时 48 分（含启动） |

计算几乎全部花在两个 2048 宽 MLP 的训练上，仿真本身只占 7%。这也是 SuperMimic 的核心思路：用可微世界模型替代大量昂贵的真实 rollout。

### 训练曲线

![LAFAN1 训练曲线](/media/lafan-curves.png)

世界模型损失在前几百次迭代内快速收敛；真实仿真中的平均回合长度在约 5,000 次迭代时接近满分 500 步，之后奖励与跟踪误差继续缓慢改善。

### 各阶段评估

评估方法见 [回放、评估与录像](../guide/evaluate.md#批量评估)：每段动作切成不重叠的 10 秒窗口（共 902 个），关闭初始扰动和推搡，保留观测噪声与域随机化，统计提前终止（失败）的窗口。

| 迭代 | 训练时长 | 成功窗口 | 有失败的动作 | 根相对 MPKPE | 末端位置误差 | 全局 MPKPE |
| --- | --- | --- | --- | --- | --- | --- |
| 5,000 | 0.66 h | 880/902（97.6%） | 15/40 | 4.8 cm | 6.9 cm | 0.385 m |
| 10,000 | 1.30 h | 896/902（99.3%） | 6/40 | 3.8 cm | 5.2 cm | 0.261 m |
| 15,000 | 1.94 h | 900/902（99.8%） | 2/40 | 3.5 cm | 4.8 cm | 0.236 m |
| 20,000 | 2.58 h | 900/902（99.8%） | 2/40 | 3.3 cm | 4.4 cm | 0.204 m |
| 25,000 | 3.19 h | **902/902（100%）** | **0/40** | 3.2 cm | 4.4 cm | 0.186 m |
| 30,000 | 3.80 h | 899/902（99.7%） | 3/40 | 3.1 cm | 4.2 cm | 0.169 m |

![各快照的成功率与跟踪误差](/media/lafan-snapshots.png)

评估中的观测噪声与域随机化是随机的，所以对最后两个快照各重复评估了 3 次：

| 快照 | 4 次评估的失败窗口数 | 出现失败的动作 |
| --- | --- | --- |
| 25,000 | 0 · 2 · 1 · 1 | fight1_subject3、run2_subject4 |
| 30,000 | 3 · 1 · 0 · 3 | fight1_subject2/3、fightAndSports1_subject1、sprint1_subject2/4、walk3_subject3 |

可以看到 2 小时之后剩下的失败都是零星、随机的，集中在格斗、冲刺这类高动态片段，每次评估出现的位置都不同。

::: info 关于“全局 MPKPE”
全局 MPKPE 统计的是机器人与参考动作在世界坐标下的距离，包含 10 秒内逐渐累积的整体位置漂移，所以数值较大；根相对 MPKPE 去掉了整体平移，更能反映姿态跟踪的精度。随着训练进行，全局漂移也在稳定下降（0.385 → 0.169 m）。
:::

## 与论文的对照

| | 论文（单张 H100） | 本机（单张 RTX 3090） |
| --- | --- | --- |
| LAFAN1 | 1.2 小时达到 0.82 归一化回报 | 1.94 小时 99.8% 窗口成功，3.19 小时 100% |
| 指标 | 归一化回报、失败动作数 | 10 秒窗口成功率、MPKPE（指标不同，不能直接比较数值） |

论文没有公开归一化回报的完整计算脚本，本仓库改用按窗口统计成功率的方式，便于在任何数据集上快速比较。从结果看，一张两代以前的消费级显卡（RTX 3090）也能在半天内完整复现 LAFAN1 上的动作跟踪效果。

## 复现本页结果

```bash
bash scripts/01_setup.sh && bash scripts/02_download_data.sh && bash scripts/03_convert_motions.sh
PRESET=lafan ITERS=30001 SNAPSHOT_EVERY=5000 bash scripts/04_train.sh
bash scripts/09_eval_snapshots.sh
```

原始文件：

- [`results/lafan-3090/run_meta.json`](https://github.com/QianYuan1437/SuperMimic-Auto/blob/main/results/lafan-3090/run_meta.json)：完整的 runner 配置；
- [`results/lafan-3090/metrics.jsonl.gz`](https://github.com/QianYuan1437/SuperMimic-Auto/blob/main/results/lafan-3090/metrics.jsonl.gz)：3,001 条训练指标；
- [`results/lafan-3090/evals/`](https://github.com/QianYuan1437/SuperMimic-Auto/tree/main/results/lafan-3090/evals)：每个快照及重复评估的 `eval.json`（含每段动作的失败窗口数与误差）；
- [`results/smoke-single/`](https://github.com/QianYuan1437/SuperMimic-Auto/tree/main/results/smoke-single)：单段动作实验。

权重文件（每个快照约 75 MB）没有放进仓库，按上面的命令约 4 小时即可重新训练得到。
