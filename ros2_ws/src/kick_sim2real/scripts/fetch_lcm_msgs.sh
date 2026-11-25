#!/usr/bin/env bash
# 从 MiRoboticsLab/loco_hl_example 拉取 LCM 生成的消息类（LCM 后端用）。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 https://github.com/MiRoboticsLab/loco_hl_example.git "$TMP/loco_hl_example"
mkdir -p "$DIR/kick_sim2real/lcm_msgs"

found=0
for f in robot_control_cmd_lcmt.py robot_control_response_lcmt.py localization_lcmt.py; do
  src="$(find "$TMP/loco_hl_example" -name "$f" | head -1 || true)"
  if [ -n "$src" ]; then
    cp "$src" "$DIR/kick_sim2real/lcm_msgs/"
    found=$((found + 1))
  else
    echo "WARN: $f 未在 loco_hl_example 中找到（可手动 lcm-gen 生成）" >&2
  fi
done
touch "$DIR/kick_sim2real/lcm_msgs/__init__.py"
echo "copied $found/3 lcm message modules -> kick_sim2real/lcm_msgs/"
