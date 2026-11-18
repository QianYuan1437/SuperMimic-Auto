# 脚本与工具一览

## scripts/ — 按步骤编号的 Shell 入口

所有脚本都可以在仓库根目录执行，路径与上游提交号统一由 `scripts/common.sh` 和 `configs/upstream.env` 管理。

| 脚本 | 作用 | 常用环境变量 / 参数 |
| --- | --- | --- |
| `00_check_env.sh` | 只读检查 GPU、驱动、显存、uv、磁盘、EGL | — |
| `01_setup.sh` | 克隆固定版本的上游代码到 `third_party/`，`uv sync --frozen`，验证 CUDA | `UV_MIRROR=<PyPI 镜像>`；`UPSTREAM_LATEST=1` 使用上游最新 main |
| `02_download_data.sh` | 下载 LAFAN1 G1 重定向 CSV 到 `data/lafan1_g1_csv/` | `HF_ENDPOINT`；`MOTIONS="a b"` 只下载部分 |
| `03_convert_motions.sh` | CSV → NPZ（50 fps）到 `data/lafan1_g1_npz/` | `--render` 预览视频；`--overwrite` |
| `04_train.sh` | 训练世界模型 + 策略 | `PRESET=single\|lafan\|custom`、`MOTION_FILE`、`TASK`、`NUM_ENVS`、`ITERS`、`RUN_NAME`、`DEPLOY=1`、`SNAPSHOT_EVERY`、`WANDB_MODE`；其余参数透传给上游 `train.py` |
| `05_play.sh` | viser / 原生查看器交互回放 | `[运行目录]`、`--viewer native`、`--motion-file` |
| `06_eval.sh` | 无头批量评估，输出 `eval.json` | `[运行目录]`、`--motion-file`、`--stride-s`、`--noise False` |
| `07_record.sh` | 离屏录制 mp4 | `[运行目录]`、`--clip`、`--start-s`、`--seconds`、`--output` |
| `08_plot.sh` | 由 `metrics.jsonl` 画训练曲线 | `[运行目录]`、`--output` |
| `09_eval_snapshots.sh` | 评估所有权重快照并汇总成表格与曲线（`snapshots_summary.json/png`） | `[运行目录]`；训练时需设置 `SNAPSHOT_EVERY` |

`[运行目录]` 可以省略（取 `third_party/supermimic/logs/$TASK/` 下最新一次），也可以传 `""` 占位后再跟其他参数。

## tools/ — Python 工具（在 supermimic 的 uv 环境中运行）

这些工具只是**包装**上游代码，不修改 `third_party/` 里的任何文件：

| 文件 | 作用 | 对应的上游实现 |
| --- | --- | --- |
| `csv_to_npz_local.py` | CSV → NPZ，保存到本地 | `mjlab.scripts.csv_to_npz`（上游会强制上传 W&B） |
| `train_local.py` | 运行上游 `train.py`，并把 `run_meta.json`、`metrics.jsonl` 写进日志目录；默认 `WANDB_MODE=offline` | `supermimic.scripts.train` |
| `local_run.py` | 从运行目录重建 env、world model、actor 并加载权重 | `supermimic.scripts.play` 中依赖 W&B 的部分 |
| `policy_utils.py` | 推理包装：组装 actor 输入、维护历史窗口 | `supermimic.scripts.viewers` / `benchmark` |
| `play_local.py` | 离线交互回放 | `supermimic.scripts.play` |
| `eval_local.py` | 按窗口批量评估成功率与跟踪误差 | `mjlab.tasks.tracking.scripts.evaluate` |
| `record_video.py` | EGL 离屏录像 | `ManagerBasedRlEnv(render_mode="rgb_array")` |
| `plot_metrics.py` | 训练曲线 | — |
| `summarize_snapshots.py` | 汇总各快照的评估结果 | — |
| `quiet_sim.py` | 关闭 MuJoCo Warp 的“迭代次数达到上限”刷屏提示（`SM_SOLVER_WARNINGS=1` 恢复） | `mjlab.sim.SimulationCfg.apply_wp_opt` |

直接调用示例：

```bash
cd third_party/supermimic
uv run python ../../tools/eval_local.py logs/SuperMimic-GenCoord-G1 --stride-s 5
```

## 上游脚本（可选）

需要 W&B 在线账号的官方脚本仍然可以直接使用：

```bash
cd third_party/supermimic
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --motion-file ~/data/x.npz
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1 --wandb-path <entity>/supermimic/<run_id>
uv run src/supermimic/scripts/benchmark.py --wandb-path <entity>/supermimic/<run_id>   # 仿真 vs 世界模型吞吐
uv run src/supermimic/scripts/train_ppo.py SuperMimic-GenCoord-G1 --motion-file ...     # PPO 基线
```
