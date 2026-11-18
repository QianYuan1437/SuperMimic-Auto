# 5. 回放、评估与录像

训练过程中权重每 `save_interval`（默认 1000）步覆盖保存一次，所以**训练还在进行时就可以评估最新权重**。下面的脚本都默认读取 `third_party/supermimic/logs/<TASK>/` 下最新的运行目录，也可以把具体目录作为第一个参数传入。

## 交互式查看器

```bash
bash scripts/05_play.sh                              # 最新一次运行
bash scripts/05_play.sh third_party/supermimic/logs/SuperMimic-GenCoord-G1/2026-11-18_18-00-00
bash scripts/05_play.sh --viewer native              # 有显示器时用 MuJoCo 原生窗口
```

默认使用 [viser](https://viser.studio) 网页查看器，浏览器打开 <http://localhost:8080>。在远程服务器上运行时，用 SSH 端口转发：`ssh -L 8080:localhost:8080 user@server`。

![viser 查看器：实体为策略控制的机器人，绿色半透明为参考动作，紫色半透明为世界模型的开环预测](/media/viser.png)

查看器由上游 `SuperMimicViserViewer` 提供，可以：

- 在下拉框中切换参考动作；
- 显示世界模型的“幽灵”机器人：从当前真实状态出发，用学到的模型开环预测接下来的若干步（`--num-steps-rollout` 控制长度），直观对比世界模型与真实物理的差别；
- 切换 motion / velocity 任务编码器。

## 批量评估

```bash
bash scripts/06_eval.sh                                     # 在训练动作上评估
bash scripts/06_eval.sh "" --motion-file "data/other/*.npz" # 在其他动作上评估
```

[`tools/eval_local.py`](https://github.com/QianYuan1437/SuperMimic-Auto/blob/main/tools/eval_local.py) 把每段动作切成 10 秒（一个 episode 长）的窗口，每个窗口分配一个并行环境，从窗口起点开始跟踪：

- 关闭随机初始扰动和推搡，保留观测噪声（与上游 `evaluate.py` 一致）；
- 运行到超时或动作结束记为**成功**；被终止条件（姿态/位置偏离过大）提前结束记为**失败**；
- 同时计算 mjlab 的跟踪误差指标。

| 指标 | 含义 | 单位 |
| --- | --- | --- |
| `success_rate` | 成功窗口比例 | — |
| `failed_windows` / `failed_clips` | 失败的窗口数 / 至少有一个窗口失败的动作数 | 个 |
| `mpkpe` | 全局身体关键点平均位置误差 | m |
| `r_mpkpe` | 以根节点为参考的关键点误差（排除整体漂移） | m |
| `joint_vel_error` | 关节速度误差 | rad/s |
| `ee_pos_error` / `ee_ori_error` | 手、脚末端的位置 / 姿态误差 | m / rad |

结果打印在终端，并保存为运行目录下的 `eval.json`（包含每段动作的失败窗口数）。

::: info 与论文指标的关系
论文在 BONES-SEED 基准上统计“失败动作数”（例如 3 小时 25/129,514）和归一化回报。本仓库的评估脚本在相同精神下按窗口统计失败，便于在 LAFAN1 等小数据集上快速比较，但数值不能与论文表格直接对比。
:::

## 比较不同训练阶段

上游每次保存都会覆盖同名权重。训练时设置 `SNAPSHOT_EVERY` 可以另存快照，事后统一评估：

```bash
SNAPSHOT_EVERY=5000 PRESET=lafan bash scripts/04_train.sh   # 生成 snapshots/it_0005000/ ...
bash scripts/09_eval_snapshots.sh                           # 逐个评估并汇总
```

汇总结果写到运行目录下的 `snapshots_summary.json` 与 `snapshots_summary.png`（成功率、跟踪误差随训练时长的变化），本机结果见 [复现记录](../results/index.md)。

## 录制视频

```bash
bash scripts/07_record.sh "" --clip dance1_subject1 --seconds 20
bash scripts/07_record.sh "" --clip fallAndGetUp1_subject1 --start-s 5 --output outputs/getup.mp4
```

使用 EGL 离屏渲染，无显示器的服务器也能录制。画面中绿色半透明的“幽灵”是参考动作（mjlab 的调试可视化），可以直观看到跟踪误差和全局漂移。

## 训练曲线

```bash
bash scripts/08_plot.sh                     # 生成运行目录下的 curves.png
```

曲线数据来自训练时写入的 `metrics.jsonl`，不依赖 W&B。
