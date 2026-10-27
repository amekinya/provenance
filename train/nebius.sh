#!/usr/bin/env bash
# Our models on Nebius AI Cloud: fine-tuning on a Serverless Job, serving on a Serverless Endpoint.
# Shared by Faithful and Provenance (one endpoint serves both). Needs the Nebius CLI with a profile.
#   BUCKET_ID=... ./train/nebius.sh train faithful-checker <DATA_URL> 0 [EPOCHS]
#   BUCKET_ID=... TOKEN=... ./train/nebius.sh endpoint            -> prints the https URL
#   ./train/nebius.sh stop|start|logs|status <ENDPOINT_ID>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NEBIUS=${NEBIUS:-nebius}
IMAGE=${IMAGE:-pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime}
PLATFORM=${PLATFORM:-gpu-l40s-a}
PRESET=${PRESET:-1gpu-8vcpu-32gb}
cmd=${1:?train|endpoint|stop|start|logs|status}; shift || true
case "$cmd" in
  train)
    NAME=${1:?name}; DATA_URL=${2:?data url}; REPORT_CLASS=${3:-1}; EPOCHS=${4:-2}
    $NEBIUS ai job create --name "train-${NAME}-$(date +%m%d-%H%M)" --image "$IMAGE" --platform "$PLATFORM" --preset "$PRESET" \
      --on-demand --volume "${BUCKET_ID:?}:/models:rw" --timeout 2h --disk-size 100Gi \
      --inject-file "$HERE/train_checker.py:/job/train.py" --inject-file "$HERE/start_train.sh:/job/start.sh" \
      --env "NAME=$NAME" --env "DATA_URL=$DATA_URL" --env "REPORT_CLASS=$REPORT_CLASS" --env "EPOCHS=$EPOCHS" --env "OUT_DIR=/models" \
      --container-command bash --args "/job/start.sh" --async --format json ;;
  endpoint)
    $NEBIUS ai endpoint create --name "${EP_NAME:-nemotron-tools}" --image "$IMAGE" --platform "$PLATFORM" --preset "$PRESET" \
      --on-demand --volume "${BUCKET_ID:?}:/models:ro" --disk-size 100Gi --container-port 8000 --auth token --token "${TOKEN:?}" \
      --inject-file "$HERE/server.py:/app/server.py" --inject-file "$HERE/start_server.sh:/app/start.sh" \
      --env "PORT=8000" --env "PRELOAD=${PRELOAD:-asr,faithful-checker,provenance-matcher,base-reranker}" \
      --container-command bash --args "/app/start.sh" --async --format json ;;
  stop|start)  $NEBIUS ai endpoint "$cmd" "${1:?endpoint id}" ;;
  logs)        $NEBIUS ai endpoint logs "${1:?endpoint id}" 2>/dev/null || $NEBIUS ai job logs "$1" ;;
  status)      $NEBIUS ai endpoint get "${1:?id}" --format json 2>/dev/null || $NEBIUS ai job get "$1" --format json ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
