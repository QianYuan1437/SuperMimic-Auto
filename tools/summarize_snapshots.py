"""汇总一次运行各个快照（snapshots/it_*/eval.json）的评估结果：表格 + 曲线。

  uv run python tools/summarize_snapshots.py <运行目录>

训练用时按 metrics.jsonl 中各段迭代的实际耗时累加（不含启动与评估时间）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

from local_run import resolve_run_dir  # noqa: E402


@dataclass
class Args:
  run_dir: tyro.conf.Positional[str]


def train_hours_at(rows: list[dict], log_interval: int) -> dict[int, float]:
  """iteration -> 累计训练小时数（采集 + WM + 策略学习时间之和）。"""
  out, total = {0: 0.0}, 0.0
  for r in rows:
    total += (
      r.get("Perf/collection_time", 0)
      + r.get("Perf/wm_learning_time", 0)
      + r.get("Perf/policy_learning_time", 0)
    )
    out[(r["_step"] + 1) * log_interval] = total / 3600
  return out


def main(args: Args) -> None:
  run_dir = resolve_run_dir(args.run_dir)
  meta = json.loads((run_dir / "run_meta.json").read_text())
  rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
  hours = train_hours_at(rows, meta["runner_cfg"]["log_interval"])

  points = []
  for ev in sorted(run_dir.glob("snapshots/it_*/eval.json")):
    it = int(ev.parent.name.split("_")[1])
    d = json.loads(ev.read_text())
    near = min(hours, key=lambda k: abs(k - it))
    points.append(
      {
        "iteration": it,
        "train_hours": round(hours[near], 3),
        "success_rate": d["success_rate"],
        "failed_windows": d["failed_windows"],
        "num_windows": d["num_windows"],
        "failed_clips": d["failed_clips"],
        "num_clips": d["num_clips"],
        "mpkpe": d["mpkpe"],
        "r_mpkpe": d["r_mpkpe"],
        "ee_pos_error": d["ee_pos_error"],
      }
    )
  if not points:
    raise SystemExit("没有找到 snapshots/it_*/eval.json，请先运行 scripts/09_eval_snapshots.sh")
  (run_dir / "snapshots_summary.json").write_text(json.dumps(points, indent=2))

  print("| 迭代 | 训练时长 (h) | 成功窗口 | 失败动作 | MPKPE (m) | 根相对 MPKPE (m) |")
  print("| --- | --- | --- | --- | --- | --- |")
  for p in points:
    print(
      f"| {p['iteration']} | {p['train_hours']:.2f} | "
      f"{p['num_windows'] - p['failed_windows']}/{p['num_windows']} "
      f"({p['success_rate'] * 100:.1f}%) | {p['failed_clips']}/{p['num_clips']} | "
      f"{p['mpkpe']:.3f} | {p['r_mpkpe']:.3f} |"
    )

  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
  x = [p["train_hours"] for p in points]
  axes[0].plot(x, [p["success_rate"] * 100 for p in points], "o-", color="#2a78d6")
  axes[0].set_ylabel("successful windows (%)")
  axes[1].plot(
    x, [p["r_mpkpe"] * 100 for p in points], "o-", color="#1baf7a", label="root-relative"
  )
  axes[1].plot(
    x, [p["ee_pos_error"] * 100 for p in points], "s--", color="#d6862a", label="end-effector"
  )
  axes[1].set_ylabel("tracking error (cm)")
  axes[1].legend()
  for ax in axes:
    ax.set_xlabel("training time (h, RTX 3090)")
    ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(run_dir / "snapshots_summary.png", dpi=130)
  print(f"[INFO] {run_dir / 'snapshots_summary.json'} / snapshots_summary.png")


if __name__ == "__main__":
  main(tyro.cli(Args))
