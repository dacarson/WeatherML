MODEL_DIR=$(dirname "$MODEL_FILE")
MODEL_BASE=$(basename "$MODEL_FILE")

docker run -it --rm --platform linux/amd64 \
  -v "$MODEL_DIR":/workspace \
  -w /workspace \
  edgetpu-x86-compiler 
