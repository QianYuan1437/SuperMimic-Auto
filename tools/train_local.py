"""离线友好的 SuperMimic 训练入口：上游 train.py + 本地元数据/指标落盘。

与 ``supermimic/scripts/train.py`` 参数完全相同，额外做两件事：

1. 在日志目录写入 ``run_meta.json``（runner 配置、任务名、动作文件、环境数），
   供 play_local.py / eval_local.py 离线加载；
2. 把每次 ``wandb.log`` 的数据同时追加到 ``metrics.jsonl``，无需 W&B 账号也能画曲线。

未设置 ``WANDB_MODE`` 时默认 offline（不需要登录 W&B）。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")

import wandb  # noqa: E402
from supermimic.runners import supermimic_runner as sr  # noqa: E402
from supermimic.scripts import train as upstream_train  # noqa: E402

_state: dict = {}
_orig_init_wandb = sr.BaseRunner._init_wandb


def _init_wandb(self, log_dir: Path, name_suffix: str = "") -> None:
  _orig_init_wandb(self, log_dir, name_suffix)
  motion_cfg = self.env.unwrapped.cfg.commands["motion"]
  meta = {
    "task_id": self.task_id,
    "runner_cfg": asdict(self.cfg),
    "motion_file": motion_cfg.motion_file,
    "num_envs": self.env.num_envs,
    "wandb_run": None if wandb.run is None else wandb.run.path,
    "argv": sys.argv[1:],
  }
  (log_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
  _state["metrics"] = open(log_dir / "metrics.jsonl", "a", buffering=1)
  # wandb.init() 会把模块级 wandb.log 重新绑定到 run.log，所以要在 init 之后再包装。
  _state["log"] = wandb.log
  wandb.log = _wandb_log
  print(f"[LOCAL] 元数据: {log_dir / 'run_meta.json'}  指标: metrics.jsonl")


def _wandb_log(row: dict, *args, **kwargs):
  f = _state.get("metrics")
  if f is not None:
    flat = {
      k: (v.item() if hasattr(v, "item") else v)
      for k, v in row.items()
      if isinstance(v, (int, float)) or hasattr(v, "item")
    }
    if "step" in kwargs:
      flat["_step"] = kwargs["step"]
    f.write(json.dumps(flat) + "\n")
  return _state["log"](row, *args, **kwargs)


sr.BaseRunner._init_wandb = _init_wandb
if os.environ["WANDB_MODE"] in ("offline", "disabled"):
  # 离线时权重只保存在本地日志目录；跳过上游的 W&B artifact 上传（需要 API key）。
  sr.log_model_artifact = lambda files, *a, **k: None

if __name__ == "__main__":
  upstream_train.main()
