# 4. 训练

## 一条命令

```bash
PRESET=single bash scripts/04_train.sh   # 单段动作 run1_subject5，验证流程
PRESET=lafan  bash scripts/04_train.sh   # 全部 40 段 LAFAN1（2.45 小时动作）
```

[`scripts/04_train.sh`](https://github.com/QianYuan1437/SuperMimic-Auto/blob/main/scripts/04_train.sh) 最终执行的是：

```bash
cd third_party/supermimic
uv run python ../../tools/train_local.py SuperMimic-GenCoord-G1 \
  --motion-file "data/lafan1_g1_npz/*.npz" --num-envs 4096 --runner.run-name lafan-1118-1800
```

`tools/train_local.py` 与上游 `src/supermimic/scripts/train.py` 的参数完全相同，只额外：

1. 默认 `WANDB_MODE=offline`，不需要 W&B 账号，并跳过 W&B 权重上传；
2. 在日志目录写 `run_meta.json`（runner 配置、任务、动作文件），供离线回放/评估；
3. 把每次记录的指标追加到 `metrics.jsonl`，可用 `scripts/08_plot.sh` 画图。

## 环境变量与参数

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PRESET` | `single` | `single` 单段 / `lafan` 全部 40 段 / `custom` 用 `MOTION_FILE` |
| `MOTION_FILE` | — | `custom` 时的 NPZ 路径，支持 glob（加引号）与逗号列表 |
| `TASK` | `SuperMimic-GenCoord-G1` | 或 `SuperMimic-MaxCoord-G1`（最大坐标表示，见下文） |
| `NUM_ENVS` | `4096` | 并行仿真环境数；显存不足时减小 |
| `ITERS` | 上游默认 1,000,000 | 最大迭代数，到达后自动结束 |
| `RUN_NAME` | `<preset>-<时间>` | 运行名 |
| `DEPLOY` | `0` | `1` 时训练只用本体感知 + 10 帧历史的可部署策略并导出 ONNX，见 [导出与部署](./deploy.md) |
| `SNAPSHOT_EVERY` | 不另存 | 每隔多少次迭代把权重另存到 `snapshots/it_XXXXXXX/`（应为 `save-interval` 的整数倍） |
| `CUDA_VISIBLE_DEVICES` | — | 指定 GPU |
| `WANDB_MODE` | `offline` | `online` 时同步到 W&B（先 `uv run wandb login`） |

其余参数原样透传给上游，例如：

```bash
bash scripts/04_train.sh --runner.policy.batch-size 2048 --runner.save-interval 500
```

全部可调参数见 [训练超参数](../reference/config.md)。

## 训练输出

```text
third_party/supermimic/logs/SuperMimic-GenCoord-G1/2026-11-18_18-00-00/
├── world_model.pt     世界模型（动力学网络 + 仿真上下文归一化）
├── actor.pt           策略（多编码器 actor）
├── critic.pt          价值网络
├── run_meta.json      本仓库写入：离线加载所需的配置
├── metrics.jsonl      本仓库写入：训练指标
└── policy_*.onnx      仅 DEPLOY=1 且动作总帧数 ≤ 15,000 时导出
```

权重每 `--runner.save-interval`（默认 1000）次迭代**覆盖**保存一次。训练没有自然的“结束”，可以随时 `Ctrl+C`，最近一次保存的权重即为结果。

## 读懂训练日志

每 10 次迭代打印一个块，重点关注：

| 字段 | 含义 | 期望趋势 |
| --- | --- | --- |
| `Steps per second` | 真实仿真每秒采集的转移数（环境数 × 迭代/秒） | 稳定 |
| `Collection / WM Learning / Policy Learning time` | 10 次迭代中仿真采集、世界模型拟合、策略 BPTT 各自耗时 | — |
| `Loss/state` | 世界模型的多步预测误差（按 Bryson 规则归一化） | 快速下降后缓慢下降 |
| `Policy/reward` | 想象 rollout 中的平均单步奖励 | 上升 |
| `Mean reward` / `Mean episode length` | 真实仿真中的回合奖励 / 长度（最长 500 步 = 10 s） | 上升 |
| `Metrics/motion/error_*` | 与参考动作的各项误差 | 下降 |
| `Episode_Termination/anchor_pos` | 因根节点偏离参考过远而终止的回合数 | 下降 |

MuJoCo Warp 的接触求解器在训练初期（机器人频繁摔倒）会每步打印 `solver iterations limit reached` / `linesearch iterations limit reached`，本机实测 16 分钟产生 13 MB 日志。这不影响训练，`tools/quiet_sim.py` 默认关掉这两类提示（其他真正的溢出警告保留）；需要看到它们时设置 `SM_SOLVER_WARNINGS=1`。

## 训练循环在做什么

一次迭代 = 4096 个环境各走一步（0.02 s）。每次迭代：

1. **采集**：当前策略在 MuJoCo Warp 中走一步，转移写入 replay buffer；
2. **按比例更新**：平均每次迭代做 $\frac{4096}{10\times2048\times8}\times16=0.4$ 次更新，每次更新包含
   - **世界模型**：从 buffer 采 10 批 × 2048 条 8 步窗口，自回归预测并最小化误差；
   - **策略**：冻结世界模型，从 buffer 状态出发想象 32 步，沿模型反向传播（BPTT）得到解析梯度，并用 GAE 拟合 critic。

所以计算量主要在第 2 步，仿真本身很便宜。原理详见 [论文精读](../paper/notes.md)。

## GenCoord 与 MaxCoord

| 任务 | 世界模型的状态表示 | 特点 |
| --- | --- | --- |
| `SuperMimic-GenCoord-G1` | 广义坐标：根节点位姿 + 29 个关节角/角速度 | 维度小（预测 35 维），训练快，默认 |
| `SuperMimic-MaxCoord-G1` | 最大坐标：每个被跟踪刚体的位置/姿态/速度 | 维度大，与奖励（按刚体计算）更直接对应 |

## 训练多久

训练速度主要取决于 GPU 的矩阵运算能力（世界模型与策略都是 2048 宽的 MLP）。本机 RTX 3090 的实测数据：

| 设置 | 迭代/秒 | 显存 |
| --- | --- | --- |
| GenCoord，4096 环境，单段动作 | 约 2.1 | 约 18.8 GB |

完整的训练曲线、各阶段评估结果与视频见 [本机复现记录](../results/index.md)。作为参照，论文在单张 H100 上 1.2 小时即可在 LAFAN1 上达到 0.82 的归一化回报。
