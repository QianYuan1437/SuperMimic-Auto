#!/usr/bin/env bash
# 依次评估一次运行的所有快照（训练时设置 SNAPSHOT_EVERY 生成），并汇总成表格与曲线。
# 用法：scripts/09_eval_snapshots.sh [运行目录]   已有 eval.json 的快照会跳过
source "$(dirname "$0")/common.sh"
TASK="${TASK:-SuperMimic-GenCoord-G1}"
RUN="${1:-$SM_DIR/logs/$TASK}"; shift || true
RUN="$(sm_run_here python -c "import sys; sys.path.insert(0, '$ROOT/tools'); from local_run import resolve_run_dir; print(resolve_run_dir('$RUN'))" 2>/dev/null | tail -1)"
[[ -d "$RUN/snapshots" ]] || die "$RUN 下没有 snapshots/（训练时请设置 SNAPSHOT_EVERY）"
for snap in "$RUN"/snapshots/it_*/; do
  [[ -f "$snap/eval.json" ]] && continue
  log "评估 $(basename "$snap")"
  sm_run_here python "$ROOT/tools/eval_local.py" "$snap" "$@"
done
sm_run_here python "$ROOT/tools/summarize_snapshots.py" "$RUN"
