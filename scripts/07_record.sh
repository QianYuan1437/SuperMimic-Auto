#!/usr/bin/env bash
# 离屏录制策略视频。用法：scripts/07_record.sh [运行目录] --clip walk1_subject1 --seconds 20
source "$(dirname "$0")/common.sh"
TASK="${TASK:-SuperMimic-GenCoord-G1}"
RUN="${1:-$SM_DIR/logs/$TASK}"; shift || true
sm_run python "$ROOT/tools/record_video.py" "$RUN" "$@"
