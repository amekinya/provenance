#!/bin/bash
# One entry point for the Nebius console (no CLI needed). The console's tiny start file downloads this script and runs it.
#   ROLE=train : fine-tune one model (NAME, DATA_URL, REPORT_CLASS, EPOCHS) and write it to the bucket at /models/NAME
#   ROLE=serve : serve every trained model in /models, plus Parakeet speech recognition, on port 8000
set -euo pipefail
RAW="https://raw.githubusercontent.com/${REPO:-amekinya/faithful}/${REF:-main}/train"
get() { python -c "import sys,urllib.request as u; u.urlretrieve(sys.argv[1], sys.argv[2])" "$RAW/$1" "$2"; }
echo "boot: role=${ROLE:-train} name=${NAME:-} gpu=$(python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
case "${ROLE:-train}" in
  train)
    get train_checker.py /tmp/train.py
    pip install -q --no-cache-dir "transformers>=5.21" accelerate safetensors
    export OUT_DIR=${OUT_DIR:-/models}
    exec python /tmp/train.py ;;
  serve)
    get server.py /tmp/server.py
    pip install -q --no-cache-dir "transformers>=5.21" accelerate safetensors fastapi "uvicorn[standard]" soundfile librosa
    export PORT=${PORT:-8000} MODELS_DIR=${MODELS_DIR:-/models}
    exec python /tmp/server.py ;;
  *) echo "ROLE must be train or serve"; exit 2 ;;
esac
