#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f /home/duanxiong/anaconda3/etc/profile.d/conda.sh ]; then
  # 兼容固定路径
  source /home/duanxiong/anaconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  # 兼容任意 Conda 安装位置
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "未找到 conda.sh，无法激活环境" >&2
  exit 1
fi

conda activate racrwhi

pushd "${REPO_ROOT}" >/dev/null

python -c "import torch; import onnx; print(torch.__version__)"
python tools/smoke_test_rwhi.py
pytest -q tests/test_rwhi_v3.py -s
python tools/export_onnx_rwhi.py

popd >/dev/null

echo "RWHI skill run PASS"
