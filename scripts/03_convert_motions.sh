#!/usr/bin/env bash
# CSV(30 fps) → NPZ(50 fps)，保存在 data/lafan1_g1_npz/。额外参数透传给 tools/csv_to_npz_local.py，
# 例如 --render 生成预览视频，--overwrite 强制重转。
source "$(dirname "$0")/common.sh"
[[ -d "$CSV_DIR" ]] || die "未找到 $CSV_DIR，请先运行 scripts/02_download_data.sh"
sm_run python "$ROOT/tools/csv_to_npz_local.py" \
  --input-dir "$CSV_DIR" --output-dir "$NPZ_DIR" "$@"
log "下一步：scripts/04_train.sh"
