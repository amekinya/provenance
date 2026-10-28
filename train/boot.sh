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
    # The bucket mount can't rename or overwrite files (safetensors saves to a temp file, then renames it), so train to
    # local disk and copy the finished model into a fresh folder <name>@<time>; DONE is written last.
    BUCKET=${OUT_DIR:-/models}
    export OUT_DIR=/tmp/out
    set +e; python /tmp/train.py; rc=$?; set -e
    if [ -d "/tmp/out/${NAME}" ]; then
      dest="${BUCKET}/${NAME}@$(date -u +%Y%m%d-%H%M%S)"
      mkdir -p "$dest"
      cp -r "/tmp/out/${NAME}/." "$dest/"
      if [ "$rc" = 0 ]; then echo "finished $(date -u)" > "$dest/DONE"; fi
      echo "copied to $dest:"; ls -la "$dest"
    fi
    exit "$rc" ;;
  serve)
    get server.py /tmp/server.py
    pip install -q --no-cache-dir "transformers>=5.21" accelerate safetensors fastapi "uvicorn[standard]" soundfile librosa
    export PORT=${PORT:-8000} MODELS_DIR=${MODELS_DIR:-/models}
    exec python /tmp/server.py ;;
  *) echo "ROLE must be train or serve"; exit 2 ;;
esac
