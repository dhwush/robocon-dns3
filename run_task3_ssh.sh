#!/usr/bin/env bash
# 在笔记本执行；识别进程、相机、雷达均在主机上运行。
set -euo pipefail
task3_host="${1:-volleyball@192.168.3.194}"
task3_mode="${2:-extend}"
case "$task3_mode" in
  basic|extend|static) ;;
  *) echo '用法：bash run_task3_ssh.sh [用户@主机IP] [basic|extend|static]' >&2; exit 2 ;;
esac
# 可用环境变量指定主机部署目录及安装了RealSense依赖的Python。
task3_dir="${TASK3_REMOTE_DIR:-robocon/robocon3}"
task3_python="${TASK3_REMOTE_PYTHON:-python3}"
printf -v task3_command 'cd -- %q && exec %q -u view_field.py %q --headless --pose-source host' \
  "$task3_dir" "$task3_python" "$task3_mode"
exec ssh -tt -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$task3_host" "$task3_command"
