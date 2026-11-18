# 训练超参数

下表整理自 `train.py SuperMimic-GenCoord-G1 --help`（上游 `supermimic@7e94b07`），列出 runner 相关的全部参数及默认值。所有参数都可以追加在 `scripts/04_train.sh` 后面，例如 `bash scripts/04_train.sh --runner.policy.batch-size 2048`。参数与论文的逐项对照（以及代码与论文不一致之处）见 [论文精读 · 超参数](../paper/notes.md)。

::: tip 写法
mjlab 关闭了 tyro 的布尔开关转换：布尔参数写成 `--runner.headless False`；元组写成 Python 语法 `--runner.actor-hidden-dims "(1024, 512, 256)"`。
:::

## 顶层

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--num-envs` | 4096 | 并行仿真环境数；同时决定 replay buffer 大小（环境数 × `buffer-size`） |
| `--motion-file` | — | 本地 NPZ（单个 / glob / 逗号分隔）；`04_train.sh` 自动设置 |
| `--registry-name` | `supermimic/wandb-registry-motions/lafan_run1_subject5` | W&B 动作 registry（作者私有）；给了 `--motion-file` 时忽略 |
| `--device` | `cuda:0` | |

## runner：训练循环与网络

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--runner.max-iterations` | 1,000,000 | 最大迭代数（1 次迭代 = 所有环境各走一步）；`ITERS` 环境变量 |
| `--runner.replay-ratio` | 16 | 更新/采集比例：每次迭代平均更新 `num_envs/(wm.num_batches·wm.batch_size·wm.horizon)×replay_ratio` = 0.4 次 |
| `--runner.buffer-size` | 1000 | 每个环境在 FIFO replay buffer 中保留的转移数 |
| `--runner.representation` | `gencoord` | 世界模型状态表示：`gencoord` / `maxcoord`（任务名含 MaxCoord 时自动切换） |
| `--runner.actor-obs` | `rep_state` | actor 输入：`rep_state` 特权状态 / `proprio` 本体感知（可部署，会导出 ONNX） |
| `--runner.history-length` | 1 | actor 观测历史帧数（>1 需要 `proprio`；论文实机设置为 10） |
| `--runner.termination-softness` | 0.05 | 想象 rollout 中可微终止的 sigmoid 软化系数 |
| `--runner.dynamics-hidden-dims` | (2048, 2048, 2048) | 世界模型 MLP 宽度 |
| `--runner.actor-hidden-dims` | (2048, 1024, 512, 256) | actor 解码器宽度 |
| `--runner.critic-hidden-dims` | (256, 128) | critic 宽度 |
| `--runner.latent-dim` | 32 | 多编码器 actor 的共享潜变量维度 |
| `--runner.latent-bc-weight` | 0.1 | 非锚定编码器向锚定编码器潜变量对齐的 BC 损失权重 |
| `--runner.sampling-mode` | `return` | 重置起点分布：`return`（集中在低回报片段）/ `uniform` / `adaptive` / `start` |
| `--runner.log-interval` | 10 | 每多少次迭代打印/记录一次 |
| `--runner.save-interval` | 1000 | 每多少次迭代覆盖保存一次权重 |
| `--runner.run-name` | 自动 | 运行名；`RUN_NAME` 环境变量 |
| `--runner.wandb-project` | `supermimic` | W&B 项目名 |
| `--runner.wandb-run-path` | — | 从 W&B 运行加载已有世界模型（热启动） |

## runner.wm：世界模型学习

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--runner.wm.lr` | 2e-4 | 学习率 |
| `--runner.wm.num-batches` | 10 | 每次更新的 batch 数 |
| `--runner.wm.batch-size` | 2048 | 每批窗口数 |
| `--runner.wm.horizon` | 8 | 多步自回归预测窗口 $k$ |
| `--runner.wm.nominal-error-root-pose-pos` / `-ori` | 0.01 / 0.02 | Bryson 规则损失归一化：根节点积分位置 / 姿态 |
| `--runner.wm.nominal-error-joint-pos` / `-vel` | 0.1 / 3.14 | GenCoord：关节位置 / 速度 |
| `--runner.wm.nominal-error-body-pos` / `-rot` / `-vel` / `-avel` | 0.05 / 0.1 / 1.0 / 3.14 | MaxCoord：刚体位置 / 姿态 / 线速度 / 角速度 |

## runner.policy：BPTT 策略学习

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--runner.policy.lr` | 2e-4 | 学习率 |
| `--runner.policy.batch-size` | 4096 | 每次更新的想象 rollout 条数（24 GB 显卡不够时可减半） |
| `--runner.policy.num-batches` | 1 | |
| `--runner.policy.horizon` | 32 | 想象视界 $H$（BPTT 长度） |
| `--runner.policy.gamma` / `lam` | 0.99 / 0.95 | 折扣与 GAE λ（actor-critic 版本） |
| `--runner.policy.survival-masking` | True | 想象中“已终止”的轨迹不再贡献奖励 |
| `--runner.policy.beta2` | 0.95 | Adam β₂（更快从梯度尖峰恢复） |
| `--runner.policy.grad-clip-value` | 0.1 | 逐元素梯度裁剪 |

## runner.return-sampler：基于回报的重置采样

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--runner.return-sampler.explore` | 0.5 | 均匀分布在所有片段上的重置比例 |
| `--runner.return-sampler.steepness` | 25.0 | 对 $V/v_{max}$ 做 softmax 的增益（越大越集中在低回报片段） |
| `--runner.return-sampler.keep` | 0.1 | 按历史关注度平方和保留的比例 |
| `--runner.return-sampler.gamma` | 0.99 | 回报折扣 |
| `--runner.return-sampler.horizon` | 20 | 每次更新使用的步数 |
| `--runner.return-sampler.memory` | 256 | 每个片段 V 估计的记忆长度 |

## runner.tasks / slot-mix：速度任务（可选）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--runner.slot-mix.motion` / `.velocity` | 1.0 / 0.0 | 环境槽位分配给动作跟踪 / 速度跟踪任务的比例，和须为 1 |
| `--runner.tasks.velocity.lin-vel-x-range` | (-1.5, 1.5) | 速度指令范围（m/s） |
| `--runner.tasks.velocity.lin-vel-y-range` | (-0.5, 0.5) | |
| `--runner.tasks.velocity.ang-vel-z-range` | (-3.14, 3.14) | rad/s |
| `--runner.tasks.velocity.window-frames` | 25 | 由参考动作计算速度指令时的平滑窗口 |
| `--runner.tasks.velocity.command-noise` | (0.5, 0.5, 0.785) | 指令噪声 |

## 环境参数

`--env.*` 下有数百个参数（奖励权重、终止阈值、域随机化范围、仿真步长等），来自 mjlab 的 G1 跟踪任务配置。查看完整列表：

```bash
cd third_party/supermimic
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --help | less
```

常用的几项：`--env.episode-length-s`（默认 10）、`--env.decimation`（默认 4，即 200 Hz 物理 / 50 Hz 控制）、`--env.seed`。
