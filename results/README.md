# results/

本机（RTX 3090 24 GB，驱动 595，Ubuntu 24.04）按本仓库脚本复现得到的原始结果，分析与图表见文档 [RTX 3090 复现记录](../docs/results/index.md)。

| 目录 | 内容 |
| --- | --- |
| `smoke-single/` | 单段动作 run1_subject5，3,000 次迭代：`run_meta.json`（配置）、`metrics.jsonl.gz`（训练指标）、`eval.json`、`curves.png` |
| `lafan-3090/` | LAFAN1 全部 40 段，30,000 次迭代（3.80 h）：配置、训练指标、`curves.png`、`snapshots_summary.{json,png}` |
| `lafan-3090/evals/` | `it_XXXXXXX.json`：各快照评估；`repeat_<快照>_<k>.json`：最后两个快照的重复评估 |

`metrics.jsonl.gz` 解压后可直接用 `tools/plot_metrics.py` 画图（需与 `run_meta.json` 放在同一目录）：

```bash
mkdir -p /tmp/lafan && cp results/lafan-3090/run_meta.json /tmp/lafan/
gunzip -c results/lafan-3090/metrics.jsonl.gz > /tmp/lafan/metrics.jsonl
bash scripts/08_plot.sh /tmp/lafan
```

权重文件未入库（每个快照约 75 MB），可按文档命令重新训练。
