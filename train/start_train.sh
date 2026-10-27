#!/bin/bash
# Entry point for the Nebius Serverless Job (injected next to train.py).
set -euo pipefail
pip install -q --no-cache-dir "transformers>=5.21" accelerate safetensors
python /job/train.py
