"""
Builds a tiny Dense(16, relu) -> Dense(4) gesture-classifier MLP (matching embedded-nn Studio's
DenseMLP architecture and default shapes: 16 Mel-bin features, 4 classes), trains it briefly on a
synthetic dataset, and converts it to a fully int8-quantized .tflite file via TFLiteConverter's
standard post-training full-integer-quantization path.

This is the reproduction script for the `dense_mlp.tflite` fixture used in
embedded-nn-tflite's accuracy-comparison test against the real TFLite Python interpreter.
"""
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

NUM_FEATURES = 16
NUM_HIDDEN = 16
NUM_CLASSES = 4
NUM_SAMPLES_PER_CLASS = 60

# Synthetic dataset: each class is a Gaussian blob in feature space, mirroring the shape of
# Studio's synthetic gesture demo (distinct centroids per class, plus noise).
centroids = np.random.uniform(-1.0, 1.0, size=(NUM_CLASSES, NUM_FEATURES)).astype(np.float32)
X = []
y = []
for c in range(NUM_CLASSES):
    pts = centroids[c] + np.random.normal(0, 0.2, size=(NUM_SAMPLES_PER_CLASS, NUM_FEATURES))
    X.append(pts)
    y.append(np.full(NUM_SAMPLES_PER_CLASS, c))
X = np.clip(np.concatenate(X).astype(np.float32), -1.0, 1.0)
y = np.concatenate(y).astype(np.int64)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(NUM_FEATURES,)),
    tf.keras.layers.Dense(NUM_HIDDEN, activation="relu"),
    tf.keras.layers.Dense(NUM_CLASSES),
])
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(X, y, epochs=30, batch_size=16, verbose=0)

loss, acc = model.evaluate(X, y, verbose=0)
print(f"float model training accuracy: {acc:.3f}")

def representative_dataset():
    for i in range(100):
        yield [X[i : i + 1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open("dense_mlp.tflite", "wb") as f:
    f.write(tflite_model)
print(f"wrote dense_mlp.tflite ({len(tflite_model)} bytes)")

# Save a handful of real quantized test inputs + the float training data range, so the Rust side
# can replay the exact same int8 inputs through both the TFLite interpreter and embedded-nn.
interp = tf.lite.Interpreter(model_path="dense_mlp.tflite")
interp.allocate_tensors()
input_details = interp.get_input_details()[0]
output_details = interp.get_output_details()[0]
in_scale, in_zero_point = input_details["quantization"]
print(f"input quant: scale={in_scale}, zero_point={in_zero_point}")

test_indices = [0, 61, 122, 183, 5, 70]
with open("test_vectors.txt", "w") as f:
    for idx in test_indices:
        sample = X[idx]
        q = np.round(sample / in_scale + in_zero_point).clip(-128, 127).astype(np.int8)
        interp.set_tensor(input_details["index"], q.reshape(1, -1))
        interp.invoke()
        out = interp.get_tensor(output_details["index"])[0]
        f.write(",".join(str(v) for v in q) + "|" + ",".join(str(v) for v in out) + f"|{y[idx]}\n")
print("wrote test_vectors.txt")
