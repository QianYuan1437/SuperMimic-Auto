#!/usr/bin/env bash
# 无头批量评估：成功率 / 失败窗口 / MPKPE 等，结果写到运行目录的 eval.json。
# 用法：scripts/06_eval.sh [运行目录] [--motion-file 'data/xxx/*.npz']
source "$(dirname "$0")/common.sh"
TASK="${TASK:-SuperMimic-GenCoord-G1}"
RUN="${1:-$SM_DIR/logs/$TASK}"; shift || true
sm_run_here python "$ROOT/tools/eval_local.py" "$RUN" "$@"
