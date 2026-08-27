# Hardware analysis (MicroFlow-style)

CSV tables comparable to MicroFlow's `analysis/` notebooks: **accuracy**, **flash weights**,
**SRAM arena**, plus a QEMU firmware-size snapshot.

| File | Contents |
| --- | --- |
| [`hardware.csv`](hardware.csv) | Host interpreter metrics for vendored MicroFlow models; QEMU Cortex-M3 sine smoke firmware; STM32WBA65 placeholder |

Refresh the host rows by running:

```
cargo test -p embedded-nn-tflite --test hardware_analysis -- --nocapture
```

If scheduler or importer numbers change, update the CSV to match. QEMU `firmware_*`
columns are a snapshot of `rust-size` on `examples/qemu-lm3s6965` (`cargo build --release`);
they include the vector table and runtime, not just weights. QEMU is **not** cycle-accurate.

On-target STM32 latency is still the GPIO/DWT procedure in
[`docs/HARDWARE_BENCHMARK_METHODOLOGY.md`](../docs/HARDWARE_BENCHMARK_METHODOLOGY.md).
