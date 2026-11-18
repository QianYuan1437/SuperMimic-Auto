#!/usr/bin/env bash
# 根据 metrics.jsonl 画训练曲线。用法：scripts/08_plot.sh [运行目录] [--output curve.png]
source "$(dirname "$0")/common.sh"
TASK="${TASK:-SuperMimic-GenCoord-G1}"
RUN="${1:-$SM_DIR/logs/$TASK}"; shift || true
sm_run_here python "$ROOT/tools/plot_metrics.py" "$RUN" "$@"
