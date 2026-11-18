# 论文精读与代码对照

> 论文：**SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**
> Qiayuan Liao, Yiyang Shao\*, Junfeng Long\*, Kevin Zakka, Roberto Horowitz, Pieter Abbeel†, Koushil Sreenath†（UC Berkeley）
> arXiv [2610.28480](https://arxiv.org/abs/2610.28480)（v1 2026-10-20，v2 2026-11-09），CoRL 2026 Oral ｜ 项目主页 <https://supermimic.github.io/>
> 官方代码：[HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)，基于 [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research)（[mujocolab/mjlab](https://github.com/mujocolab/mjlab) 的 fork）
>
> 本文的数值都来自论文 v2 正文和附录（下文记作“论文”），代码默认值都核对过本地仓库 `supermimic@7e94b07`（2026-11-15）和 `mjlab_research@6ea3e57`（2026-11-17）。下文路径默认相对 `supermimic/src/supermimic/`。

## 1. 问题与动机

- **背景**：要训练通用、表现力强的人形全身控制器，目前的主流做法是跟踪大规模人类动捕数据（DeepMimic → PHC → BeyondMimic → SONIC）。动作库越大，控制器越强。
- **瓶颈**：现有管线都用无模型 RL（PPO），只有两条路：
  1. 先训练大量单动作专家，再蒸馏成一个策略。蒸馏有损（BeyondMimic 走的是这条路）；
  2. 直接靠集群算力硬训一个通用策略。SONIC 用 128 张 L40 训练约 168 h，约 21,000 GPU·h。
- **洞察**：所有参考动作虽然不同，但都由**同一个人形机器人的刚体动力学**执行。无模型 RL 没有显式利用这个共享结构，每个动作都要从带噪的仿真 rollout 里重新估计“动作的后果”。
- **方案**：从仿真器日志拟合**一个可微世界模型**，覆盖所有动作。然后在短的想象 rollout 上做 BPTT，给策略提供**解析梯度**。梯度来自学到的模型，而不是对仿真器求导，所以可以继续使用带非光滑接触、本身不可微的 MuJoCo Warp。

## 2. 方法

### 2.1 训练循环（论文 §3.1，Fig. 2，Algorithm A.1）

每次迭代按顺序做三件事，世界模型和策略**在同一个循环里联合训练**，不分阶段：

1. **Collect**：当前策略 $\pi_\theta$ 在 MuJoCo Warp 中采集转移，写入固定大小的 FIFO replay buffer $\mathcal{B}$。环境的重置起点按 return-based reset sampling 抽取。
2. **Fit WM**：从 $\mathcal{B}$ 采样 $k$ 步窗口，在日志动作上自回归 rollout $f_\omega$，最小化 $\mathcal{L}_{WM}$。
3. **Improve policy**：冻结 $\omega$，从 buffer 状态出发（Dyna 风格）在 $f_\omega$ 中想象 $H$ 步，用 BPTT 更新 $\theta$，同时拟合 critic $V_\psi$。

因为动力学与采集数据的策略无关，旧策略产生的转移依然可以用来拟合世界模型。

代码中的对应关系：`runners/supermimic_runner.py:SuperMimicRunner.run()` 是主循环。每一步先调用 `step()` 采集，再由 `Ratio` 决定这一步做几次“WM 更新 + 策略更新”：

$$\text{updates/iter} = \frac{\mathrm{num\_envs}}{\mathrm{wm.num\_batches}\cdot\mathrm{wm.batch\_size}\cdot\text{wm.horizon}}\times\mathrm{replay\_ratio} = \frac{4096}{10\cdot 2048\cdot 8}\times 16 = 0.4$$

这正好对应论文 Table A.5 的 update-to-data ratio 0.4，以及“每次 WM 更新配 1 次策略更新”。

### 2.2 POMDP 与回报（§3.1）

- 策略是确定性的：$a_t=\pi_\theta(o_t,g_t)$，$a_t$ 参数化归一化的关节设定点 $q^{sp}=\alpha a_t+q^0$。PD 增益较低，所以设定点实际上是力矩的代理。
- 回报使用**软存活权重**：$c_0=1,\ c_{t+1}=c_t(1-d_{t+1})$，$J(\theta)=\mathbb{E}[\sum_t\gamma^t c_t r_{t+1}]$。仿真器给出硬终止；世界模型输出的终止概率 $\hat d_t\in[0,1]$ 让存活权重保持可微。
- 观测分解为 $o_t=\bar o_t+\eta_t$。世界模型只预测无噪观测 $\bar o_t$，传感器噪声 $\eta_t$ 在模型外部加入。
- 论文说采集时可以加动作噪声 $\epsilon_t\sim\mathcal N(0,0.1^2)$，只用来拓宽 WM 的拟合分布，并且是可选的。**最新代码已经删除了这项噪声**（commit `7e94b07` “Drop the action noise in data collection”）。commit message 给出的理由是：用 Muon 加 10 帧历史时，策略会依赖这项噪声。在 bones_full、300k 迭代、2 GPU 的设置下，干净基准上的失败数从 2,315 降到 1。

### 2.3 可微世界模型（§3.2）

$$\hat s_{t+1}=f_\omega(\hat s_t,a_t,\xi),\quad \hat r_{t+1}=r_\omega(\hat s_t,a_t,g_t),\quad \hat{\bar o}_t=g_\omega(\hat s_t,\xi),\quad \hat d_t=d_\omega(\hat s_t,g_t)$$

- **确定性**：仿真器的随机性来源已知，所以模型不建模随机性，而是以每个回合的 **sim context** $\xi$（摩擦、关节零位偏置、躯干质心偏移）为条件。
- 每个 head 要么学习得到，要么解析计算。解析 head 不产生损失项。**公开代码里只有动力学 $f_\omega$ 是学出来的**，奖励、观测、终止都是解析计算（见下表）。
- 损失：在 $k=8$ 步自回归窗口上计算，

$$\mathcal{L}_{WM}(\omega)=\mathbb{E}_\mathcal{B}\sum_{j=1}^{k}\Big[\|\hat s_j-s_j\|^2+(\hat r_j-r_j)^2+\|\hat{\bar o}_j-\bar o_j\|^2+\mathrm{BCE}(\hat d_j,d_j)\Big]$$

  代码中只剩状态项。按 **Bryson 法则**分组加权，每组权重为 $1/\mathrm{nominal\_error}^2$（`learners/wm_learner.py:WorldModelLearner._compute_losses`）。
- 网络（`models.py:Dynamics`）：先对状态和动作做 EmpiricalNormalization，再拼上 $\xi$，送入 MLP（Linear→LayerNorm→GELU）×3，隐藏层宽 2048。网络输出**加速度级增量**，由 `representation.apply_prediction` 用半隐式积分得到下一状态。积分时位姿增量乘以 `substep_coeff = (decimation+1)/(2·decimation)`，用来匹配仿真器 4 个子步的合成方式。

#### GenCoord 与 MaxCoord（论文未提及，属于代码中的两种状态表示）

论文正文**没有出现** GenCoord/MaxCoord 这两个词。它们是官方代码提供的两种世界模型状态表示，对应两个任务 ID：

| | `SuperMimic-GenCoord-G1`（广义坐标） | `SuperMimic-MaxCoord-G1`（最大坐标，参考 ControlVAE） |
|---|---|---|
| 类 | `representations.py:GenCoordRepresentation` | `representations.py:MaxCoordRepresentation` |
| full state | root pos(3)+quat(4)+$q_j$(29)+root v/ω(6)+$\dot q_j$(29) = 71 维 | 14 个跟踪刚体 ×（pos 3 + quat 4 + v 3 + ω 3）= 182 维 |
| rep state | $[z, g_b, v_b, \omega_b, q_j, \dot q_j]$ | 各刚体在根坐标系下的 pos、rot6d、v、ω，加根高度和 up 向量 |
| 网络输出 | $[\Delta v_b,\Delta\omega_b,\Delta\dot q_j]$（6+29） | 每个刚体的 $[\Delta v,\Delta\omega]$（14×6） |
| 刚体位姿 / 奖励 | 正运动学 `self.fk`（来自 MotionLoader） | 直接得到刚体位姿；关节量用解析 IK `utils/analytic_ik.py:AnalyticIK` 反解 |
| 损失分组 | root_pos/root_ori/joint_pos/joint_vel | root_pos/root_ori/body_pos/body_rot/body_vel/body_avel |

`scripts/train.py` 会根据任务名自动设置 `runner.representation`。`runners/run_config.py:check_representation_matches` 保证回放时使用的表示与训练时一致。

### 2.4 通过世界模型学习策略（§3.3）

- 冻结 $\omega$，在 $H=32$ 步想象 rollout 上优化 $\pi_\theta$ 和 $V_\psi$。每条 rollout 从 buffer 状态 $s_0$ 出发，并带上它记录的 $\xi$。每一步把传感器噪声重新加到 $\hat{\bar o}_t$ 上，再交给策略。
- 目标函数：

$$\hat J(\theta)=\mathbb{E}\Big[\sum_{t=0}^{H-1}\gamma^t\hat c_t\hat r_{t+1}+\gamma^H\hat c_H V_\psi(\hat s_H,g_H,\xi)\Big]$$

- BPTT 通过世界模型对 $\hat s_t$ 和 $a_t$ 的雅可比得到一阶梯度。和 PPO 的似然比估计相比，短视界下方差低得多。
- Critic 用 GAE（$\lambda=0.95$）拟合，只用想象转移，目标在每条 rollout 上算一次后固定：
  $\hat R_t=\hat r_{t+1}+\gamma(1-\hat d_{t+1})[(1-\lambda)V_{t+1}+\lambda\hat R_{t+1}]$，$\hat R_H=V_H$。
- 设计上类似 SHAC，但用软存活权重 $\hat c_t$ 代替硬终止，而且**没有** EMA 目标网络。论文强调 critic 在受扰恢复时有作用：去掉 critic 后名义跟踪回报差不多，但策略在 0.5 m/s 的推力下就站不稳，4 m/s 时会摔倒。

代码：`learners/policy_learner.py`
- `BpttPolicyLearner._train_batch`：想象 rollout，累加 `gamma**t * survival * reward`，`survival *= (1 - term_prob)`，再用 `-ret + aux_loss` 反传。梯度先做全局范数 1.0 裁剪，再做逐元素 0.1 裁剪。
- `ActorCriticPolicyLearner._bootstrap`：末端价值乘以 $\gamma^H\hat c_H$，并截断到 $V_{max}=r_{max}/(1-\gamma)$。
- `ActorCriticPolicyLearner._fit_critic` + `gae_returns`：5 个 epoch，每个 epoch 4 个 minibatch。
- `SuperMimicRunnerCfg.policy` 的默认值是 `ActorCriticPolicyLearnerCfg`（带 critic）。无 critic 的 `BpttPolicyLearnerCfg` 对应论文的消融。但 mjlab 的 `TYRO_FLAGS` 启用了 `tyro.conf.AvoidSubcommands`，没法从命令行切换 union 类型，要切换只能改代码里的默认值。

### 2.5 多编码器 actor 与下游接口（附录 B）

$$z=\mathrm{enc}_{\theta,i}(g)\in\mathcal Z,\qquad a=\mathrm{dec}_\theta(z,o)$$

每种目标接口（完整参考动作、速度指令等）各有一个编码器，所有接口共享解码器。latent 为 32 维，经 tanh 有界。训练目标是 $\max_\theta\mathbb E_{i\sim p(\mathcal I)}[\hat J_i(\theta)]$，所有接口共用同一个 $f_\omega$。

- 代码：`models.py:MultiEncoderAE`。batch 按 `slot_mix` 切成连续的块，每块走对应任务的编码器。第一个任务（motion）是 anchor，其他编码器通过 `latent_bc_weight=0.1` 的 latent-BC 损失向 anchor 的 latent 靠拢（anchor 一侧 stop-grad）。对应论文 G.1 的 “MSE weight 0.1”。
- 任务定义：`tasks/task_spec.py`（`make_motion_task`、`make_velocity_task`）。
- **和论文的差异**（以公开代码为准）：
  - 默认 `slot_mix={"motion": 1.0, "velocity": 0.0}`，即只训练动作跟踪。
  - 公开的 `MultiTaskWorldModelEnv` 写明 “Reward is motion-tracking for every slot”，也就是只实现了**蒸馏版**的速度编码器。论文 §4.5 中用速度跟踪奖励（Table A.6）训练的 **RL 速度编码器未包含在公开代码里**。
  - 速度编码器在代码中与动作编码器共用 `actor_hidden_dims=(2048,1024,512,256)`，论文 Table A.4 写的是 (256,256)。速度范围默认 $v_x\in[-1.5,1.5]$、$v_y\in[-0.5,0.5]$、$\omega_z\in[-3.14,3.14]$，论文 G.1 给的是 $[-1.5,2]$、$\pm1.32$、$\pm3.35$。

### 2.6 Return-based reset sampling（附录 D.1）

仿真回合重置到动作库中的某个时刻，并偏向策略当前回报低的位置。具体做法：把时间轴分 bin，每个 bin 维护一个折扣回报的滑动估计 $G_b$，重置分布按下式混合：

$$p_b=\underbrace{0.5\cdot\tfrac{1}{N}}_{\text{explore}}+\underbrace{0.4\cdot\mathrm{softmax}_b(-25\,G_b/G_{max})}_{\text{focus}}+\underbrace{0.1\cdot\text{keep}_b}_{\text{过去 focus 权重}}$$

- 代码：`runners/reset_sampling.py:ReturnSampler`（`ReturnSamplerCfg`：`gamma=0.99, horizon=20, memory=256, steepness=25.0, explore=0.5, keep=0.1`）。keep 项正比于各 bin 历史 focus 的**平方和**（`effort += focus²`）。
- 接入方式：`SuperMimicRunner._configure_reset_sampling` 和 `step()` 调用 `ReturnSampler.record()`，结果经 `MotionCommand.set_sampling_distribution` 推给 mjlab，mjlab 侧使用 `sampling_mode="external"`（`mjlab_research/src/mjlab/tasks/tracking/mdp/commands.py`）。`train.py` 会自动把 `runner.sampling_mode="return"` 映射成 env 的 `"external"`。
- **bin 宽度不一致**：论文写 1 s 一个 bin，`mjlab_research` 的 `MotionCommand` 实际是 `bin_count = time_step_total // (0.2/step_dt) + 1`，即 **0.2 s** 一个 bin。

### 2.7 Sim context 与域随机化（附录 A，Table A.3）

| 参数 | 分布 | 代码 |
|---|---|---|
| 足底摩擦 $\mu$ | $U[0.3,1.2]$，所有足部 geom 共享同一个值 | `sim_context.py:FootFrictionSpec`（1 维） |
| 关节零位偏置 $\Delta q^0_j$ | $U[-0.01,0.01]$ rad | `EncoderBiasSpec`（29 维） |
| 躯干质心偏移 | $\Delta x\sim U[\pm0.025]$，$\Delta y,\Delta z\sim U[\pm0.05]$ m | `CoMOffsetSpec`（`base_com`，3 维） |

- `SimContext.from_env` 从 env 中启用的 DR 事件自动发现这些量，并按 DR 区间归一化到 $[-1,1]$。$\xi$ 同时输入给 Dynamics 和 critic（critic 的输入是 `rep_state + wm_goal + sim_ctx`，属于特权信息）。
- `tasks/env_cfgs.py:gencoord_env_cfg` 只保留 `foot_friction / base_com / encoder_bias` 三个事件，**去掉了 BeyondMimic 的推力扰动**。原因是确定性世界模型无法刻画推力造成的状态跳变；论文认为只随机化物理参数已经足够迁移到实机。

### 2.8 终止模型

- 仿真端的终止条件：根高度误差 $|e_{p,z}|>0.25$ m，或倾斜误差 $e_{tilt}>0.8$（直立参考下约 78°）。$e_{tilt}$ 是参考坐标系和机器人根坐标系中重力方向 z 分量之差的绝对值。**与 BeyondMimic 不同，不对末端执行器偏差终止**（`env_cfgs.py` 中 `terminations.pop("ee_body_pos")`）。
- 世界模型端：论文的式 (1) 用 BCE 学习 $d_\omega$；**公开代码改成了解析的软终止**，见 `envs/world_model_env.py:WorldModelEnv._compute_term_prob`：

$$\hat d=\sigma\Big(\frac{\max(e_z/0.25,\ e_{tilt}/0.8)-1}{\mathrm{termination\_softness}}\Big),\quad \mathrm{termination\_softness}=0.05$$

### 2.9 观测、动作与奖励（附录 A）

- 本体观测：$q-q^0$（29，噪声 $U[\pm0.01]$）、$\dot q$（29，$U[\pm0.5]$）、IMU 角速度（3，$U[\pm0.2]$）、投影重力（3，$U[\pm0.05]$）、上一步动作（29）。论文中策略输入**最近 10 帧**观测。
  代码见 `representations.py:PROPRIO_NOISE_HALF_WIDTHS` 和 `env_cfgs.py:_sensor_actor_group`。
- 目标：$K=10$ 帧未来参考。代码为 `NUM_GOAL_FRAMES=10`、`GOAL_FRAME_STRIDE=5`，帧间隔约 0.1 s，前瞻约 0.9 s；论文没有写 stride。actor 看到的目标不含参考锚点的相对位置 `rel_pos_b`，这一项只给世界模型和 critic（`actor_goal_dim`）。
- 奖励：沿用 BeyondMimic 的跟踪奖励（Table A.2），包括刚体位置、朝向、线速度、角速度各 1.0，根位置、根朝向各 0.5，动作平滑 −0.1，关节限位 −2（原为 −10），并**去掉了自碰撞惩罚**。论文给出的理由是：自碰撞惩罚本来是防止随机策略借贴合身体来压低噪声，而这里的策略在想象中是确定性的，这种 reward hacking 拿不到好处。所有项乘以 $\Delta t=0.02$ s。
  代码中世界模型端的奖励在 `utils/tracking_reward.py:tracking_reward`（解析计算，目标刚体重新锚定到预测根的 xy+yaw），仿真端在 `env_cfgs.py`（`rewards.pop("self_collisions")`、`joint_limit.weight=-2.0`）。
- **⚠ actor 的默认输入**：代码默认 `actor_obs="rep_state"`（特权的世界模型状态）、`history_length=1`。只有 `actor_obs="proprio"` 时才会导出可部署的 ONNX，见 `SuperMimicRunner.run`：

  ```python
  if self._representation.actor_obs_mode == "proprio" and self.motion.time_step_total <= 15_000:
      export_actor_as_onnx(...)
  ```

  要复现论文中可上实机的设置，需要加上 `--runner.actor_obs proprio --runner.history_length 10`（参数写法沿用官方 README 的 `--runner.run_name` 风格；`history_length>1` 时强制要求 proprio）。这一条是根据代码和论文对照得出的推断，官方 README 没有写明。另外，动作帧数超过 15,000（约 5 min）时会跳过 ONNX 导出，大数据集需要另行导出。

## 3. 论文组件 → 代码位置对照表

| 论文组件 | 论文出处 | 代码位置（`file:function/class`） |
|---|---|---|
| 训练主循环（采集 / 拟合 WM / 改进策略） | §3.1, Alg. A.1 | `runners/supermimic_runner.py:SuperMimicRunner.run`, `.step` |
| 更新/数据比 0.4 | Table A.5 | `supermimic_runner.py:Ratio` + `SuperMimicRunnerCfg.replay_ratio=16` |
| Replay buffer（FIFO，每个 env 1000 步） | Fig. 2 | `replay_buffer.py:ReplayBuffer`；`BaseRunnerCfg.buffer_size=1000`；字段见 `BaseRunner._buffer_schema` |
| 世界模型构建 | §3.2 | `supermimic_runner.py:build_world_model` |
| 动力学网络 $f_\omega$ | §3.2, Table A.4 | `models.py:Dynamics`, `build_mlp` |
| GenCoord / MaxCoord 状态表示与积分 | （仅代码） | `representations.py:GenCoordRepresentation/MaxCoordRepresentation.apply_prediction`；`utils/gencoord_helpers.py`、`utils/maxcoord_helpers.py`、`utils/analytic_ik.py` |
| $k$ 步自回归 WM 损失 | Eq. 1 | `learners/wm_learner.py:WorldModelLearner._rollout`, `._compute_losses`（Bryson 权重） |
| 观测 head $g_\omega$（解析） | §3.2 | `representations.py:*.compute_wm_obs`, `compute_actor_sensor` |
| 奖励 head $r_\omega$（解析） | §3.3, Table A.2 | `representations.py:compute_wm_reward_from_reference` → `utils/tracking_reward.py:tracking_reward` |
| 终止 head $d_\omega$（解析软终止） | §3.1/3.2 | `envs/world_model_env.py:WorldModelEnv._compute_term_prob` |
| 想象环境（从 buffer 出发的 Dyna rollout） | §3.3 | `envs/world_model_env.py:WorldModelEnv.reset/step/_wm_step/sample_initial_states`；`envs/multi_task_world_model_env.py` |
| BPTT 策略更新 + 软存活权重 | Eq. 2 | `learners/policy_learner.py:BpttPolicyLearner._train_batch` |
| Critic 自举 + GAE | Eq. 2–3 | `policy_learner.py:ActorCriticPolicyLearner._bootstrap`, `._fit_critic`, `gae_returns` |
| 多编码器 actor / 共享解码器 / latent 对齐 | App. B, G.1 | `models.py:MultiEncoderAE`；`tasks/task_spec.py` |
| Return-based reset sampling | App. D.1 | `runners/reset_sampling.py:ReturnSampler`；`mjlab_research/.../tracking/mdp/commands.py:MotionCommand.set_sampling_distribution` |
| Sim context $\xi$ | §3.2, Table A.3 | `sim_context.py:SimContext`, `FootFrictionSpec`, `CoMOffsetSpec`, `EncoderBiasSpec` |
| 环境配置（去掉推力、自碰撞和 EE 终止，joint_limit=−2） | App. A | `tasks/env_cfgs.py:gencoord_env_cfg`, `maxcoord_env_cfg`；任务注册见 `tasks/__init__.py` |
| 传感器噪声 $\eta_t$ | Table A.1 | `representations.py:PROPRIO_NOISE_HALF_WIDTHS`, `add_proprio_noise` |
| 优化器 / 梯度裁剪 / CUDA Graph | Table A.5 | `learners/learner_utils.py:build_optimizers`, `_optimize_step`, `CapturedUpdate` |
| PPO-WM 基线（在 WM 上跑 PPO） | §4.2, Fig. 5b | `runners/ppo_runner.py:PpoRunner`；`scripts/train_ppo.py` |
| WM 与仿真吞吐基准 | Fig. 5a | `scripts/benchmark.py` |
| ONNX 导出（部署用） | §4.4 | `runners/exporter.py:export_actor_as_onnx`（opset 18，附带 mjlab 跟踪元数据） |
| 动作数据转换（CSV → npz → W&B registry） | App. E | `mjlab_research/src/mjlab/scripts/csv_to_npz.py`（默认 30 → 50 fps） |
| 训练评估 | App. F | `mjlab_research/src/mjlab/tasks/tracking/scripts/evaluate.py`（论文的 129,514 动作基准脚本未单独公开） |

## 4. 关键超参数：论文 vs 代码默认值

论文数值取自 Table A.5、A.4。代码默认值逐项核对过源码。

| 超参数 | 论文 | 代码字段 = 默认值 | 备注 |
|---|---|---|---|
| WM 窗口长度 $k$ | 8 | `WorldModelLearnerCfg.horizon = 8` | `_rollout` 只对后 $k-1$ 步打分 |
| WM 学习率 $\alpha_\omega$ | 1 GPU 时 $2\times10^{-4}$；2 GPU 时 $\sqrt2\times2\times10^{-4}$ | `WorldModelLearnerCfg.lr = 2e-4` | |
| WM batch | 每次更新 10 批 × 2048 窗口 | `num_batches = 10`, `batch_size = 2048` | |
| WM 梯度裁剪 | 全局范数 1.0 | `_optimize_step(..., max_norm=1.0)` | |
| WM 损失权重 | 未给出 | `nominal_error_root_pose_pos=0.01`, `_ori=0.02`, `joint_pos=0.1`, `joint_vel=3.14`；MaxCoord：`body_pos=0.05`, `body_rot=0.1`, `body_vel=1.0`, `body_avel=3.14` | 权重 $=1/e_{nom}^2$ |
| 动力学 MLP | (2048, 2048, 2048) | `BaseRunnerCfg.dynamics_hidden_dims = (2048, 2048, 2048)` | |
| 想象视界 $H$ | 32 | `BpttPolicyLearnerCfg.horizon = 32` | |
| 策略 batch | 每次更新 4096 条想象 rollout | `batch_size = 4096`, `num_batches = 1` | |
| 策略 Adam β | (0.9, 0.95) | `beta2 = 0.95` | |
| 策略梯度裁剪 | 全局范数 1.0，再逐元素 0.1 | `grad_clip_value = 0.1` | |
| 折扣 $\gamma$ | 0.99 | `ActorCriticPolicyLearnerCfg.gamma = 0.99` | 纯 BPTT 版默认 `gamma=1.0` |
| GAE $\lambda$ | 0.95 | `lam = 0.95` | |
| Critic 训练 | 5 epoch × 4 minibatch | `_fit_critic` 中写死 | |
| 软存活 | 有 | `survival_masking = True`；`termination_softness = 0.05` | |
| Actor | 编码器、解码器均为 (2048, 1024, 512, 256)，latent 32 | `actor_hidden_dims = (2048,1024,512,256)`, `latent_dim = 32` | |
| Critic | (256, 128) | `critic_hidden_dims = (256, 128)` | |
| Latent 对齐权重 | 0.1 | `latent_bc_weight = 1e-1` | |
| 仿真环境数 / GPU | 4096 | `TrainConfig.num_envs = 4096`（`scripts/train.py`） | PPO-Sim 基线用 8192 |
| Replay buffer | 每个 env 1000 步 | `buffer_size = 1000` | |
| 动作噪声 | $\mathcal N(0,0.1^2)$ | **已删除**（commit `7e94b07`） | |
| 更新/数据比 | 0.4 | `replay_ratio = 16`（结合上面的公式得到 0.4） | |
| Reset sampling | 50% / 40% softmax(−25·) / 10% | `ReturnSamplerCfg(explore=0.5, steepness=25.0, keep=0.1, gamma=0.99, horizon=20, memory=256)` | bin 宽度：论文 1 s，代码 0.2 s |
| 学习率调度 | 余弦衰减到 0.03×（250k 迭代内），之后恒定 | **公开代码中没有调度器**（搜不到 scheduler/cosine） | |
| 训练长度 | 320,000 迭代 | `SuperMimicRunnerCfg.max_iterations = 1000000` | 需要用 `--runner.max_iterations` 自行设置 |
| 多 GPU | 数据并行，梯度取平均 | **公开的 `train.py` 只支持单 GPU**（没有 DDP 或 torchrun 逻辑） | |
| 优化器 | Adam | `build_optimizers`：MLP 隐层 Linear 权重用 **Muon**，其余参数用 fused Adam | Muon 是 2026-10-26 的 commit `0e5ff6b` 引入的 |
| 精度 | TF32 | `mjlab.utils.torch.configure_torch_backends` | |
| 仿真 | 50 Hz 控制 = 4 个 5 ms 子步 | `mjlab_research/.../tracking_env_cfg.py`：`timestep=0.005`, `decimation=4`, `episode_length_s=10.0` | |
| 默认动作数据 | — | `TrainConfig.registry_name = "supermimic/wandb-registry-motions/lafan_run1_subject5"` | 作者自己的 W&B registry，外部用户一般需要改用 `--motion-file` 加载本地 npz |

## 5. 实验与关键数字

所有成本都在 Lambda 按需实例的 **NVIDIA H100 80GB SXM5** 上测得。作者说跑过多个随机种子、训练非常稳定，所以图中没有画种子方差。

**归一化回报**：整个回合逐步奖励之和（不折扣），除以这个和可能的最大值。提前终止的回合因此得分低。

### 5.1 与 SONIC 对比（§4.1，Fig. 3a）

| 指标 | SONIC（发布的 checkpoint） | SuperMimic |
|---|---|---|
| 训练算法 | 仿真中的 PPO | 在学到的世界模型上做 BPTT |
| 硬件与时长 | 128 张 L40 × 168 h（7 天），约 21,000 GPU·h | 1 张 H100 × 3 h |
| 动作数据 | 611 h | 264.56 h（BONES-SEED 公开子集，约少 2.3×） |
| 归一化回报 | **0.90** | 3 h 内超过 0.90 |
| 129,514 动作基准上的失败数 | **179**（v1.1 checkpoint 为 239） | 1 GPU × 3 h：**25**；2 GPU × 9.8 h：**0** |

- 加速比：wall-clock 约 56×（论文保守地写作 >50×），GPU·h 约 7200×（保守写作 >1000×）。
- “3 h” 的口径：从 3 h 起，之后的每个 checkpoint 在回报和失败数两项上都比 SONIC 好，领先幅度至少是“SONIC 与 SuperMimic 最佳结果之差”的一半。3 h 包含完整的训练循环，不含 checkpoint 评估。
- 硬件差异：H100 的 fp32 峰值是 L40 的 0.74×，tensor-core 峰值是 5.5×。论文认为 2.3× 的数据差和至多 5.5× 的峰值算力差，解释不了 1000× 的 GPU·h 差距。
- 失败分组（Table A.8）：SONIC 在整组动作上全部失败，包括全部 44 个空翻、全部 22 个侧滚、一套舞蹈的全部 8 个动作，另有 24 个立定跳中的 15 个失败。SuperMimic 3 h 的 25 个失败里，有 14 个集中在前 5 个难动作组（共 195 个动作），而且没有任何一组的失败数多于 SONIC。
- 注意：基准动作就是训练动作，所以这个数字衡量的是**覆盖度**，不是泛化能力。

### 5.2 与 PPO-Sim 的受控对比（LAFAN1，40 个片段，2.45 h，Fig. 4a）

PPO-Sim 与 SuperMimic 使用相同的动作、观测、奖励、终止、DR、网络规模和评测协议，超参数用 BeyondMimic 的 PPO 配置，每张 GPU 8192 个环境。

- SuperMimic 在 **1 张 GPU 上 1.2 h 达到回报 0.82**。
- PPO-Sim 在 18 h 预算内没有达到 0.82。最终回报：1 GPU 为 0.60，4 GPU 为 0.70，8 GPU 为 0.72。

### 5.3 世界模型的三个性质（§4.2）

1. **一个模型摊销整个动作库**：数据集从 1 h 增加到 264 h（子集为 1/7/40/264 h），8 步预测误差始终在 **0.007–0.011**（只算状态，按预设 nominal error 归一化，在策略 rollout 上在线测得）。到达目标回报所需的 GPU·h 增长缓慢（Fig. 3b，论文没有给出具体数值）。
2. **想象 rollout 更快**：在实际使用的 2k–16k 并行环境下，世界模型比 MuJoCo Warp 快 **10× 以上**；在 64k 并行环境时快 **62×**，此时 MuJoCo Warp 的吞吐已经饱和。（Fig. 5a 上还标注了 11× 和 26×。）
3. **一阶梯度省样本**：在 LAFAN1 的 walk/run/sprint 动作上，SuperMimic 达到 PPO-WM 最佳训练回报所用的想象步数约少 **64×**。PPO-WM 的最佳配置是 16k×32。

### 5.4 视界与 critic 的消融（§4.3，Fig. 4b,c）

- 没有 critic 时，$H\le16$ 训练失败。有 critic 时，$H=16$ 就能达到无 critic $H=32$ 的效果；$H=32$ 到达 0.82 最快。
- $H$ 从 64 增加到 128，到达 0.82 的时间差不多变成 3 倍，最终回报也更低。**$H=128$ 是 H100 显存能容纳的最长视界。**
- 静态 T-pose 下施加 0.5–4 m/s 的速度推力：有 critic 的策略在 0.5–2.4 s 内稳定下来；无 critic 的策略在 0.5 m/s 时就一直踉跄，4 m/s 时摔倒。

### 5.5 实机（§4.4，Fig. 6）

**同一个**通用跟踪策略零样本部署到 **Unitree G1**，没有针对单个动作做调参。演示的动作包括跳步（skipping jump）、侧滚（side roll，躯干和四肢着地）、90° 空翻、原地 360° 空翻。论文**没有**给出实机的定量指标，也没有描述部署硬件和软件栈。

### 5.6 下游速度接口（§4.5，App. G，属于初步研究）

- RL 编码器（用速度跟踪奖励训练）与蒸馏编码器对比，在 672 个阶跃响应上测试：RL 编码器在 **650** 个上误差更低，蒸馏编码器在 **15** 个上更低。中位相对稳态误差分别为 **0.16** 和 **0.49**。
- 蒸馏编码器在 7 个“平移 + ≥2.8 rad/s 转向”的指令上每次都摔倒，这些区域在动作库中几乎没有数据。
- 两个编码器各训练 100,000 迭代，每个只有一个种子。

### 5.7 数据：BONES-SEED（附录 E）

- 来源：Bones Studio 发布的 **BONES-SEED**（Skeletal Everyday Embodiment Dataset），论文引用的地址是 <https://huggingface.co/datasets/bones-studio/seed>。使用的是其中已经重定向到 G1 的动作集。
- 过滤：原始 142,220 个片段，经名称过滤（去掉 stair/ladder/box/chair/sit/climb 等，共 12,553 个）剩 129,667 个；再经可行性过滤（MuJoCo 正运动学检测悬空支撑、穿地、双脚悬浮，去掉 153 个）最终保留 **129,514** 个片段，即 264.56 h。
- 可行性过滤的阈值：
  - 悬空支撑：最低刚体高度 $z_{min}>0.20$ m 持续 ≥0.6 s；
  - 穿地：$z_{min}<-0.12$ m，或超过 80% 的时间 $<-0.08$ m；
  - 双脚悬浮：直立且几乎静止时，双脚离地 >0.02 m 持续 ≥0.5 s。
- **公开代码不包含这套过滤脚本**，过滤后的片段列表也没有发布。

## 6. 局限性（论文 §6）与复现提示

论文列出的局限：

1. **多接口训练研究不足**：速度奖励只更新速度编码器，不更新共享解码器。因此“并行训练”等价于先训练跟踪器、再训练速度编码器。如果让速度奖励也更新解码器，并行训练是否有帮助，还没有验证。
2. **只用仿真数据**：世界模型只从仿真转移学习。用实机数据微调、或利用可微性做系统辨识，都留作后续工作。
3. **只在平地上**：不包含地形和物体交互，作者说这方面的工作正在进行。

从代码对照中额外发现的复现注意点（不是论文原文）：

- 公开代码中**没有学习率调度和多 GPU 训练**。论文 2 GPU × 9.8 h 达到零失败的结果，不能直接用公开的 `train.py` 复现。
- 优化器已经从 Adam 换成 Muon 加 Adam，采集时的动作噪声也删掉了。按 commit `7e94b07` 的说法，这样在 bones_full 上 300k 迭代后，干净基准只失败 1 个片段，但这是作者在 commit message 中的说法，没有写进论文。
- 默认的 actor 输入是特权状态（`rep_state`），部署时需要改成 `proprio` 加 10 帧历史。
- 默认动作数据来自作者私有的 W&B registry，外部用户需要自己用 `csv_to_npz.py` 生成 npz，再通过 `--motion-file` 传入（多个文件用逗号分隔，也可以用通配符）。
- **显存与时长**：论文只给出了 H100 80GB 上的数字（$H=128$ 是能容纳的上限），**没有给出显存占用或消费级显卡的数据**。按 buffer schema 粗算（本文估算）：
  - GenCoord 每条转移约 160 个 float（state 71 + action 29 + prev_action 29 + sim_ctx 33），4096 env × 1000 步约 **2.7 GB**；
  - MaxCoord（state 182 维）约 **4.5 GB**；
  - 开启 proprio 历史时再加约 1.5 GB；
  - 显存的大头是 $H=32$、4096 条 rollout 的 BPTT 激活。

  在 24 GB 显卡上，可以优先减小 `--runner.policy.batch_size`、`--num_envs` 或 `--runner.buffer_size`，或者在保留 critic 的前提下把 $H$ 降到 16（论文显示有 critic 时 $H=16$ 仍然可用）。社区在 24 GB 显卡上的实测数据（例如 realmimic 把策略 batch 降到 2048 后，用 RTX 4090 训练了 10.5 h）见 [community.md](./community.md)。
