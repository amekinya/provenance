#!/bin/bash
# Entry point for the Nebius Serverless Endpoint (injected next to server.py).
set -euo pipefail
pip install -q --no-cache-dir "transformers>=5.21" accelerate safetensors fastapi "uvicorn[standard]" soundfile librosa
exec python /app/server.py
