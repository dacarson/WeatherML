MODEL_FILE=$(realpath "$1")
MODEL_DIR=$(dirname "$MODEL_FILE")
MODEL_BASE=$(basename "$MODEL_FILE")

docker run --rm --platform linux/amd64 \
  -v "$MODEL_DIR":/workspace \
  -w /workspace \
  edgetpu-x86-compiler \
  /opt/edgetpu_compiler/edgetpu_compiler -a -s "$MODEL_BASE"
