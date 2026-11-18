#!/usr/bin/env bash
# 训练 world model + 跟踪策略。日志/权重写到 third_party/supermimic/logs/<TASK>/<时间戳>/。
#
# 预设（PRESET 环境变量）：
#   single  单段动作 run1_subject5（上游默认动作，最快看到效果）       [默认]
#   lafan   全部 40 段 LAFAN1 动作（约 2.45 小时动作数据）
#   custom  使用 MOTION_FILE 指定的 npz（glob 或逗号分隔）
# 其他环境变量：
#   TASK=SuperMimic-GenCoord-G1 | SuperMimic-MaxCoord-G1
#   NUM_ENVS=4096   ITERS=<最大迭代数>   RUN_NAME=<名字>   CUDA_VISIBLE_DEVICES=0
#   DEPLOY=1        训练可部署到真机的策略：只用本体感知 + 10 帧历史（论文实机设置），会导出 ONNX
#   WANDB_MODE=online 同步到 W&B（需先 uv run wandb login），默认 offline
#   SNAPSHOT_EVERY=5000  每 5000 次迭代另存一份权重快照到 snapshots/（默认不另存）
# 其余命令行参数原样传给上游 train.py，例如：--runner.replay-ratio 8
source "$(dirname "$0")/common.sh"

PRESET="${PRESET:-single}"
TASK="${TASK:-SuperMimic-GenCoord-G1}"
NUM_ENVS="${NUM_ENVS:-4096}"
case "$PRESET" in
  single) MOTION_FILE="$NPZ_DIR/run1_subject5.npz" ;;
  lafan)  MOTION_FILE="$NPZ_DIR/*.npz" ;;
  custom) [[ -n "${MOTION_FILE:-}" ]] || die "PRESET=custom 需要设置 MOTION_FILE" ;;
  *) die "未知 PRESET=$PRESET" ;;
esac
compgen -G "$MOTION_FILE" >/dev/null || [[ "$MOTION_FILE" == *,* ]] \
  || die "找不到动作文件 $MOTION_FILE，请先运行 scripts/03_convert_motions.sh"

args=("$TASK" --motion-file "$MOTION_FILE" --num-envs "$NUM_ENVS")
args+=(--runner.run-name "${RUN_NAME:-$PRESET-$(date +%m%d-%H%M)}")
[[ -n "${ITERS:-}" ]] && args+=(--runner.max-iterations "$ITERS")
[[ "${DEPLOY:-0}" == 1 ]] && args+=(--runner.actor-obs proprio --runner.history-length 10)

log "任务 $TASK | 预设 $PRESET | 动作 $MOTION_FILE | 并行环境 $NUM_ENVS"
log "按 Ctrl+C 可随时停止，权重每 save_interval(默认 1000) 步保存一次"
sm_run python "$ROOT/tools/train_local.py" "${args[@]}" "$@"
