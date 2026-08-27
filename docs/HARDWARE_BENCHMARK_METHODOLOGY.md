# Phase E: Benchmark Methodology & Results

This documents what was actually measured for `embedded-nn` against a real, externally-produced
model, what tooling was used, and what is intentionally left as documented methodology because it
requires physical hardware or reference software not available in this environment.

The model under test throughout is `crates/embedded-nn-tflite/fixtures/dense_mlp.tflite`: a real
`Dense(16, relu) -> Dense(4)` classifier, trained and fully int8-quantized by a genuine TensorFlow
install (see `crates/embedded-nn-tflite/fixtures/README.md` for the reproduction script). It is
imported via `embedded-nn-tflite`, compiled via `embedded-nn-codegen`, and executed via
`embedded-nn`'s runtime kernels — the complete real pipeline, not a synthetic hand-built graph.

## 1. Accuracy — real, automated, passing

`crates/embedded-nn-tflite/tests/real_model_accuracy.rs` imports the fixture, generates
`#![no_std]` Rust code, compiles it against `embedded-nn` in a scratch crate, and diffs
`predict()`'s output against real `tf.lite.Interpreter` (`BUILTIN_REF` resolver, not the XNNPACK
delegate — see the fixture README for why that distinction matters) output for 6 real quantized
input vectors.

**Result: 6/6 bit-exact matches** against genuine TFLite reference output.

This process caught a real, previously-shipped bug: `embedded-nn-codegen`'s ReLU activation
clamp was hardcoded to `Activation::new(0, i8::MAX)`, which is only correct for symmetric
(zero-point == 0) quantization. Real TFLite models are commonly asymmetrically quantized; with a
nonzero output zero-point the clamp silently zeroed out most ReLU-activated channels instead of
clamping to the zero-point. Fixed in `emit_rust.rs`'s `activation_expr` (now takes the output
tensor's `QuantParams` and clamps at `zero_point` instead of the literal `0`), with a fast unit
regression test (`test_relu_activation_clamp_uses_output_zero_point_not_literal_zero` in
`emit_rust.rs`) in addition to the end-to-end fixture test.

## 2. Static memory footprint — real, via `enn profile`

```
$ enn import --tflite crates/embedded-nn-tflite/fixtures/dense_mlp.tflite --out dense_mlp.json
$ enn profile --model dense_mlp.json

Total Layers:          2
Total Weights (Flash): 560 bytes
Peak Arena (SRAM):     36 bytes (zero heap allocation)
```

These numbers come directly from the generated code's static arrays (`FLASH_WEIGHTS_BYTES`,
`ARENA_SIZE_BYTES`) and `ArenaScheduler`'s interval-coloring allocation plan — not an estimate.
For reference, the source `.tflite` FlatBuffer is 2624 bytes, but that figure includes FlatBuffer
schema/table overhead, operator metadata, and per-tensor quantization records that don't exist at
all in embedded-nn's generated code, so it isn't a fair flash-size comparison — 560 bytes of
`static` weight/bias/multiplier/shift arrays is the actual number that lands in flash.

## 3. Compiled code size — real, cross-compiled to a Cortex-M target

Built the generated model as a `#![no_std]` `staticlib` for `thumbv7em-none-eabihf` (a real Cortex-M4F
target, `opt-level = "z"`, `lto = true`, `panic = "abort"`, `codegen-units = 1`):

```
$ rustup target add thumbv7em-none-eabihf   # already installed in this environment
$ cargo build --release --target thumbv7em-none-eabihf   # crate-type = ["staticlib"]
$ llvm-size libembedded_size_check.a
```

With fat LTO collapsing the crate + `embedded-nn` into a single compilation unit, the resulting
object measured **~1150 bytes of `.text`+`.rodata` combined** (code + the 560 bytes of weight
tables together) before final linking. This is a pre-link, single-object-file number — it doesn't
include whatever a real firmware's linker pulls in for the interrupt vector table, reset handler,
or any `compiler_builtins` helper functions actually referenced (e.g. `memcpy` if the target
requires it) — a full firmware image would need a target-specific `memory.x` linker script and
`-C link-arg=-Wl,--gc-sections`, which is genuinely target/board-specific and out of scope for a
general library. For an all-integer, no-float, no-alloc leaf crate like this, that extra
overhead is typically tens to a few hundred bytes, not a different order of magnitude.

## 4. Host-side compute latency — real, both Criterion and a direct timing loop

`crates/embedded-nn/benches/kernels.rs` (Criterion, `cargo bench -p embedded-nn`), run on this
development machine (Apple Silicon, native `aarch64-apple-darwin`, **not** representative of
Cortex-M timing — see §5):

| Benchmark | Time |
|---|---|
| `fully_connected_s8_16x16` (per-tensor quant) | ~32.5 ns |
| `fully_connected_per_channel_s8_16x16` (per-channel quant) | ~45.2 ns |
| `convolve_1_x_n_s8_16w_3k_8c` | ~909 ns |
| `softmax_s8_4classes` | ~50.1 ns |
| `requantize` (single fixed-point rescale) | ~0.70 ns |

Separately, timing the *actual* generated `DenseMlpNet::predict()` (both FC layers, the full
real imported model, 200,000 iterations after a 1,000-iteration warm-up, `opt-level = 3`, `lto =
true`): **~242 ns per `predict()` call**, consistent with the per-kernel Criterion numbers above
(two per-channel FC calls dominate; ~45 ns × 2 plus loop/dispatch overhead lands in that range).

## 5. On-target (real MCU) latency — methodology only, not executed here

This environment has no attached hardware, so the following is documented as the procedure to
follow on real hardware rather than executed:

1. Flash the generated code (via the `thumbv7em-none-eabihf` build path validated in §3) onto a
   Cortex-M4F dev board (e.g. STM32F4 Discovery, nRF52840 DK), wired up with the existing
   `embedded-nn-live` USB/UART bridge (`crates/embedded-nn-live/src/host.rs`) for host
   communication.
2. Toggle a GPIO pin (or use a hardware timer / DWT cycle counter, `cortex-m::peripheral::DWT`)
   immediately before and after the `predict()` call inside the firmware's main loop.
3. Capture the GPIO toggle on a logic analyzer (or read back the DWT cycle count over UART/probe-rs)
   across a batch of N calls (N ≥ 1000) to get a stable min/median/max, matching the same
   statistical rigor Criterion applies on the host.
4. Alternatively, `probe-rs`'s `cargo embed` + its RTT-based timing hooks (`probe-rs run
   --measure`) can automate steps 2–3 without a logic analyzer, at the cost of RTT's own small
   overhead — acceptable for a relative (not absolute) comparison across model architectures/
   quantization modes.
5. Repeat for a Cortex-M0+ target (e.g. `thumbv6m-none-eabi`, already installed in this
   environment and known to build — see the `add_conv1d_layer`/`SVDF` code paths, which have no
   hardware FPU dependency since everything is integer-only) to get a lower-end latency bound;
   the CMSIS-NN-style fixed-point kernels in `embedded-nn` have no floating-point instructions in
   their hot path, so M0+ vs M4F should differ mainly in clock speed and DSP-extension
   instruction availability (M4F's `SMULL`/`SMMLA` alternatives), not in code correctness.
6. Record results as `board,mcu,clock_mhz,model,quant_mode,latency_ns_min,median,max` rows —
   mirroring microflow-rs's `analysis/` CSV convention — once real hardware is available.

Checked-in host and QEMU snapshots live in [`analysis/hardware.csv`](../analysis/hardware.csv)
(MicroFlow `sine` / `speech` / `person_detect` import metrics, plus the LM3S6965 QEMU sine
firmware). The STM32WBA65 row is a placeholder until a board run fills latency.

No accuracy-vs-TFLite-Micro on-target comparison is included here because embedded-nn has no
TFLite Micro reference build wired into this workspace; §1's TFLite Python reference comparison
is the available substitute and is already bit-exact.
