"""根据 train_local.py 写出的 metrics.jsonl 画训练曲线（不依赖 W&B）。

uv run python tools/plot_metrics.py logs/SuperMimic-GenCoord-G1 [--output curves.png]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

from local_run import resolve_run_dir  # noqa: E402

# 标题用英文，避免没有中文字体的机器上显示为方框。
DEFAULT_KEYS = [
  ("Train/mean_reward", "mean episode reward"),
  ("Train/mean_episode_length", "mean episode length (steps)"),
  ("Policy/reward", "imagined rollout reward"),
  ("Loss/state", "world-model state loss"),
  ("Metrics/motion/error_body_pos", "body position error (m)"),
  ("Episode_Termination/anchor_pos", "terminations: anchor_pos"),
]


@dataclass
class Args:
  run_dir: tyro.conf.Positional[str]
  output: str | None = None
  keys: list[str] = field(default_factory=list)
  """要画的指标名（默认画一组常用指标）。"""


def main(args: Args) -> None:
  run_dir = resolve_run_dir(args.run_dir)
  rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
  if not rows:
    raise SystemExit("metrics.jsonl 为空")
  log_interval = json.loads((run_dir / "run_meta.json").read_text())["runner_cfg"]["log_interval"]
  present = {k for r in rows for k in r}
  keys = [(k, k) for k in args.keys] or [(k, t) for k, t in DEFAULT_KEYS if k in present]
  cols = 3
  nrows = (len(keys) + cols - 1) // cols
  fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.0 * nrows), squeeze=False)
  for ax, (key, title) in zip(axes.flat, keys, strict=False):
    xs = [r.get("_step", i) * log_interval for i, r in enumerate(rows) if key in r]
    ys = [r[key] for r in rows if key in r]
    ax.plot(xs, ys, lw=1.2, color="#2a78d6")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
  for ax in list(axes.flat)[len(keys) :]:
    ax.axis("off")
  fig.tight_layout()
  out = Path(args.output) if args.output else run_dir / "curves.png"
  fig.savefig(out, dpi=130)
  print(f"[INFO] 曲线: {out}")


if __name__ == "__main__":
  main(tyro.cli(Args))
