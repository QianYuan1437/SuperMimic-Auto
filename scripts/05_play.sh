#!/usr/bin/env bash
# 交互式回放（viser 网页查看器，浏览器打开 http://localhost:8080）。
# 用法：scripts/05_play.sh [运行目录]   默认取 logs/<TASK> 下最新一次运行
source "$(dirname "$0")/common.sh"
TASK="${TASK:-SuperMimic-GenCoord-G1}"
RUN="${1:-$SM_DIR/logs/$TASK}"; shift || true
sm_run python "$ROOT/tools/play_local.py" "$RUN" "$@"
