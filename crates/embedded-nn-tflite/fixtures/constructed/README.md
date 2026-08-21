# Constructed end-to-end fixtures

These four files are deterministic, spec-compliant TFLite FlatBuffers built directly with this
repository's generated schema bindings. They were **not produced by TensorFlow**. TensorFlow is
not available in the fixture-generation environment (Python 3.14), so no `BUILTIN_REF` provenance
is claimed for these expected vectors.

Regenerate them with:

```sh
cargo run -p embedded-nn-tflite --features fixture-generation \
  --example generate_constructed_fixtures
```

The execution tests are in
`crates/embedded-nn-macros/tests/tflite_end_to_end.rs`. All four files pass through the direct
`.tflite` proc-macro path, generated Rust code, arena scheduling, and real integer kernels.

- `sine_fc_int8.tflite`: one-neuron INT8 FC linear approximation of sine over the central
  `[-pi/2, pi/2]` interval. Its scales make the FC preserve quantized input codes, so both the
  integer and `predict_f32` vectors are independently `q_out = q_in`.
- `tinyconv_int8.tflite`: speech-scale `49x40x1` TinyConv-style chain:
  `CONV_2D -> MAX_POOL_2D -> RESHAPE -> FULLY_CONNECTED -> SOFTMAX`. For a real-zero input, all
  four logits are equal, so the expected signed softmax code is `64 - 128 = -64` per class.
- `uint8_fc.tflite`: UINT8 FC containing endpoint weight bytes `0` and `255`. The importer maps
  bytes and zero-points by subtracting 128, then generated code executes only s8 kernels.
- `add_transpose_int8.tflite`: two-input affine ADD with fused ReLU6 followed by rank-2
  transpose. Inputs are selected on exact quantization steps, making expected sums and row-major
  permutation directly calculable.

The genuine TensorFlow-produced `../dense_mlp.tflite` and its recorded `BUILTIN_REF` vectors remain
the authoritative external-runtime comparison and are intentionally separate from these fixtures.
