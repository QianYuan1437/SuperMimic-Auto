# 6. 导出与真机部署

::: danger 安全第一
在真机上运行学习得到的策略有摔倒和伤人的风险。务必先做 sim2sim 验证；上机时使用吊架，并让一人手持遥控器随时进入阻尼/急停模式。本节只介绍官方代码提供的导出能力和社区的部署方案，本仓库**没有**在真机上验证过。
:::

## 训练可部署的策略

官方代码的默认设置 `actor_obs="rep_state"` 让 actor 直接读取世界模型状态（根节点位置、速度等特权信息），这些量在真机上测不到，**不能直接部署**。要训练可部署的策略，actor 只能看本体感知（IMU 角速度、重力投影、关节位置/速度、上一帧动作）：

```bash
DEPLOY=1 PRESET=lafan bash scripts/04_train.sh
# 等价于给上游 train.py 追加：
#   --runner.actor-obs proprio --runner.history-length 10
```

`history_length=10` 让 actor 看到最近 10 帧本体感知，用来弥补缺失的特权信息，与论文实机设置一致（详见 [论文精读 §2.9](../paper/notes.md)）。

## ONNX 导出

`actor_obs=proprio` 时，训练循环每次保存权重都会自动调用 `runners/exporter.py:export_actor_as_onnx`，在运行目录下写出：

| 文件 | 输入 → 输出 | 用途 |
| --- | --- | --- |
| `policy_motion.onnx` | `(obs, time_step)` → `(action, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w)` | 动作跟踪。参考动作数组作为常量打包在模型里，按 `time_step` 取出当前参考帧 |
| `policy_velocity.onnx` | `obs` → `action` | 摇杆速度控制（需要训练时启用 velocity 任务） |
| `policy_velocity_replay.onnx` | 同 motion | sim2real 检查用：速度指令由参考动作离线算出 |

ONNX 的元数据（`onnx.load(...).metadata_props`）里附带部署需要的全部常量，部署端应当从这里读取，而不是手写。下面是本机用 `DEPLOY=1` 导出的 `policy_motion.onnx` 的实际内容（`python -c "import onnx; m=onnx.load('policy_motion.onnx'); print(m.metadata_props)"`）：

| 项 | 本机导出结果 |
| --- | --- |
| 输入 | `obs` [1, 1570]，`time_step` [1, 1] |
| 输出 | `actions` [1, 29]，`joint_pos`/`joint_vel` [1, 29]，`body_pos_w`/`body_lin_vel_w`/`body_ang_vel_w` [1, 14, 3]，`body_quat_w` [1, 14, 4] |
| `observation_names` | `projected_gravity, base_ang_vel, joint_pos, joint_vel, actions, command, motion_anchor_ori_b` |
| `observation_history_lengths` | 前 5 项（本体感知，3+3+29+29+29 = 93 维）各 10 帧 = 930 维；`command` 与 `motion_anchor_ori_b` 共 640 维，各 1 帧 |
| `joint_names` | 29 个关节，从 `left_hip_pitch_joint` 开始（与 Unitree G1 29-DoF 一致） |
| `joint_stiffness` / `joint_damping` | PD 增益，例如髋 pitch 40.18 / 2.56，膝 99.10 / 6.31 |
| `default_joint_pos` / `action_scale` | 关节目标 = `default_joint_pos + action_scale * action` |
| `anchor_body_name` / `body_names` | `pelvis` / 14 个被跟踪刚体 |

导出的三个文件大小约 30~47 MB（单段 run1_subject5，参考动作数组打包在 `policy_motion.onnx` 中）。

::: warning 大数据集不会自动导出
为了避免把巨大的动作库打包进 ONNX，参考动作总帧数超过 15,000（50 fps 下约 5 分钟）时会**跳过**导出。LAFAN1 全集（约 44 万帧）属于这种情况，需要只用要部署的那几段动作重新导出（例如用 `PRESET=custom MOTION_FILE=...` 训练或微调一个小动作集）。
:::

## 部署链路（社区方案）

官方仓库不包含实机部署代码。社区里比较完整、可以参考的方案：

| 方案 | 机器人 | 说明 |
| --- | --- | --- |
| [NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real) | G1 29-DoF | 完整链路：ONNX → `sim2sim.py`（延迟、负重、摩擦扰动验收）→ `g1_runner_bridge.py`（unitree_sdk2）。中文文档 |
| [KyrieYoung926/supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy) | G1 29-DoF | 轻量：unitree_sdk2_python + CycloneDDS + onnxruntime，可先在 MuJoCo 中运行 |
| [RoboJuDo-Plus（supermimic 分支）](https://github.com/chifongip/RoboJuDo-Plus/tree/supermimic) | G1 23-DoF | 支持 loco / mimic 切换 |

社区方案的观测布局、PD 增益与官方代码不一定一致，接入官方导出的 ONNX 前要逐项核对元数据。推荐流程：

1. **sim2sim**：在 MuJoCo（CPU 版）里用与真机相同的 50 Hz 控制循环、PD 增益和观测拼接方式运行 ONNX，确认能稳定跟踪；
2. **加扰动**：控制延迟 10~20 ms、质量 ±10%、摩擦系数变化，确认仍不摔倒；
3. **吊架上机**：先运行站立或慢走动作，再逐步尝试高动态动作。

更多社区经验见 [社区复现与扩展 · Sim2Real](../paper/community.md)。
