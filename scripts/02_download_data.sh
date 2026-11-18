#!/usr/bin/env bash
# 下载 LAFAN1 → Unitree G1 重定向后的动作 CSV（40 段，约 85 MB）。
# 国内网络可先 export HF_ENDPOINT=https://hf-mirror.com
# 只下载部分动作：MOTIONS="run1_subject5 walk1_subject1" bash scripts/02_download_data.sh
source "$(dirname "$0")/common.sh"

HF="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p "$CSV_DIR"

if [[ -n "${MOTIONS:-}" ]]; then
  names=($MOTIONS)
else
  log "从 $HF 获取文件列表"
  mapfile -t names < <(curl -fsSL "$HF/api/datasets/$LAFAN1_HF_REPO/tree/main/g1" \
    | python3 -c 'import json,sys; [print(x["path"].split("/")[-1][:-4]) for x in json.load(sys.stdin) if x["path"].endswith(".csv")]')
fi
(( ${#names[@]} > 0 )) || die "文件列表为空，检查网络或设置 HF_ENDPOINT"

for n in "${names[@]}"; do
  out="$CSV_DIR/$n.csv"
  if [[ -s "$out" ]]; then continue; fi
  log "下载 $n.csv"
  curl -fsSL --retry 3 -o "$out.part" "$HF/datasets/$LAFAN1_HF_REPO/resolve/main/g1/$n.csv"
  mv "$out.part" "$out"
done
log "完成：$(ls "$CSV_DIR" | wc -l) 个 CSV 位于 $CSV_DIR"
log "下一步：scripts/03_convert_motions.sh"
