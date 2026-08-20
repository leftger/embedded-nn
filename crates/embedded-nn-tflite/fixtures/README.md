# Real TFLite fixture: `dense_mlp.tflite`

Unlike the hand-built FlatBuffer fixtures in `src/lib.rs`'s `#[cfg(test)] mod fixtures`, this is a
**genuine** fully-int8-quantized model produced by a real TensorFlow install
(`tf.lite.TFLiteConverter`, full integer post-training quantization), used as an end-to-end
ground-truth check: import it, generate code, run `predict()`, and diff against real
`tf.lite.Interpreter` output.

- `build_model.py` — reproduction script. Trains a `Dense(16, relu) -> Dense(4)` classifier
  (matching Studio's `DenseMLP` default shapes) on a synthetic dataset, converts it via
  `TFLiteConverter` with `TFLITE_BUILTINS_INT8` + `inference_input_type/output_type = tf.int8`,
  and writes `dense_mlp.tflite` plus `test_vectors.txt`.
- `test_vectors.txt` — 6 lines, each `input_i8_csv|output_i8_csv|true_class`. The `output_i8_csv`
  values are the **real** quantized output of TFLite's reference kernels (not the XNNPACK
  delegate — see note below) for the corresponding input, i.e. ground truth to diff against.
- `dense_mlp.tflite` — the model itself.

To regenerate (requires a TensorFlow install; Python 3.12 recommended — TF wheels lag behind
newer Python releases):

```
python3 -m venv venv && ./venv/bin/pip install tensorflow
./venv/bin/python3 build_model.py
```

## Important gotcha: XNNPACK delegate hides intermediate tensors

`tf.lite.Interpreter`'s default XNNPACK delegate fuses FullyConnected+bias+activation into a
single op and does not populate the graph's original per-op debug tensors (`...MatMul`,
`...BiasAdd`, `...Relu` entries from `get_tensor_details()`). Reading those tensors via
`interp.get_tensor(idx)` after `invoke()` returns **stale/aliased buffer contents** (e.g. it may
silently return another tensor's *constant* weight or bias buffer instead of an error), which
looks like a real per-invocation activation value but isn't. If you need genuine intermediate
activations for debugging, either only trust the model's actual input/output tensors, or force
the reference kernels:

```python
tf.lite.Interpreter(
    model_path="dense_mlp.tflite",
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
)
```

This is how a real bug was caught in `embedded-nn`: comparing embedded-nn's generated
`predict()` output against genuine TFLite reference-kernel output (not the input/output-only
values, which the XNNPACK path still gets right) surfaced that
`embedded-nn-codegen`'s ReLU activation clamp was hardcoded to `Activation::new(0, i8::MAX)`,
ignoring the output tensor's zero-point — wrong for any asymmetrically-quantized (nonzero
zero-point) output, which is the common case for real TFLite models. Fixed in
`emit_rust.rs`'s `activation_expr` to clamp at `output_zero_point` instead of `0`. See
`tests/real_model_accuracy.rs` for the regression test built from this fixture.
