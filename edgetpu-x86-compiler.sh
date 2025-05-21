#!/bin/sh

MODEL_FILE="$1"

if [ -z "$MODEL_FILE" ]; then
  echo "Usage: $0 <model_file.tflite>"
  exit 1
fi

docker run --rm --platform linux/amd64 \
  -v ~/Documents/GitHub/WeatherML/workspace:/workspace \
  -w /workspace \
  edgetpu-x86-compiler \
  /opt/edgetpu_compiler/edgetpu_compiler -a -s "$MODEL_FILE"
