import tensorflow as tf
from export_edge_tpu_model import build_model, SEQ_LEN, n_features

model = build_model()
model.load_weights("./checkpoints/model_100.weights.h5")

for layer in model.layers:
    w = layer.get_weights()
    if w:
        print(layer.name, [v.shape for v in w])
