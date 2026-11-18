# 3. 准备动作数据

SuperMimic 训练的是**动作跟踪**策略：给定一段参考动作（重定向到 Unitree G1 的关节轨迹），让机器人在物理仿真里跟着做。所以训练前需要先准备参考动作。

## 数据格式

| 阶段 | 格式 | 内容 |
| --- | --- | --- |
| 重定向 CSV | 每行 36 列，30 fps | `base_pos(3), base_quat_xyzw(4), joint_pos(29)`，Unitree 的广义坐标约定 |
| mjlab NPZ | 50 fps | `root_pos_w`, `root_quat_w`(wxyz), `root_lin_vel_w`, `root_ang_vel_w`, `joint_pos`, `joint_vel`, `joint_names`, `fps` |

训练时的控制频率是 50 Hz，所以要把 CSV 插值到 50 fps，同时用有限差分算出速度。**必须使用 mjlab 的转换器**：NPZ 里的 `joint_names` 必须与机器人模型的关节名逐一相同，其他框架生成的 NPZ 会在加载时报错。

## 第一步：下载 LAFAN1（G1 版）

```bash
bash scripts/02_download_data.sh
# 只下载部分动作
MOTIONS="run1_subject5 dance1_subject1" bash scripts/02_download_data.sh
# 国内网络
HF_ENDPOINT=https://hf-mirror.com bash scripts/02_download_data.sh
```

数据来自 Hugging Face 数据集 [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 的 `g1/` 目录，共 40 段（约 85 MB），覆盖走、跑、冲刺、跳跃、舞蹈、格斗、跌倒起身等类型。上游训练脚本的默认动作 `lafan_run1_subject5` 就是其中的 `run1_subject5`。

## 第二步：转换为 NPZ

```bash
bash scripts/03_convert_motions.sh            # 输出到 data/lafan1_g1_npz/
bash scripts/03_convert_motions.sh --render   # 额外渲染每段动作的参考预览视频
```

转换器 [`tools/csv_to_npz_local.py`](https://github.com/QianYuan1437/SuperMimic-Auto/blob/main/tools/csv_to_npz_local.py) 复用了上游 `mjlab.scripts.csv_to_npz.MotionLoader`（插值、四元数 slerp、速度计算完全一致），区别只是**结果保存到本地**，不上传 W&B。40 段转换后共 441,140 帧（50 fps），约 2.45 小时动作，NPZ 约 120 MB，转换耗时约 1 分钟。

## 使用自己的动作

任何按上表 CSV 格式重定向到 G1（29 自由度）的动作都可以：

```bash
uv run --project third_party/supermimic python tools/csv_to_npz_local.py \
  --input-dir my_csv --output-dir data/my_npz --input-fps 30
PRESET=custom MOTION_FILE="data/my_npz/*.npz" bash scripts/04_train.sh
```

`MOTION_FILE` 支持单个文件、glob（需加引号）和逗号分隔的列表。

## 更大的数据集：BONES-SEED

论文主实验用的是 **BONES-SEED**（SONIC 611 小时数据中公开的 264.56 小时子集，约 13 万段动作）。

- 获取：Hugging Face [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed)，需要申请（学术用途，或年收入 100 万美元以下的公司），G1 部分为 `g1.tar.gz`（约 23.5 GB）。
- 帧率是 **120 fps**，转换时要加 `--input-fps 120`。
- 段数很多时，用上游的 `mjlab/scripts/pack_motions.py` 打包成一个目录，避免加载时打开十几万个文件；`MotionLoader` 可以直接读取打包目录。
- 动作库会常驻显存，13 万段在 24 GB 显卡上需要相应减小并行环境数和 batch，详见 [社区复现](../paper/community.md)。
