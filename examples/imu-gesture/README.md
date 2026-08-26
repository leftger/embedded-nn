# 6-DOF IMU Gesture Recognition (Compile-Time Embedding)

Demonstrates real-time 6-DOF / 9-DOF motion gesture classification using `embedded-nn`'s procedural macro `#[embedded_nn_model]`.

---

## How It Works

The `#[embedded_nn_model]` macro imports your `.tflite` or `.json` model directly at compile time, runs the interval-colored SRAM scheduler, generates zero-allocation Rust inference kernels, and embeds static weights into Flash memory:

```rust
use embedded_nn_macros::embedded_nn_model;

#[embedded_nn_model("models/gesture_model.json")]
pub struct GestureClassifier;

fn main() {
    // Exact statically calculated SRAM arena
    let mut arena = [0u8; GestureClassifier::ARENA_SIZE];
    let imu_sample = [12i8, -4, 28, 5, -15, 30, -8, 14, -20, 18, 10, -12, 25, -9, 7, -2];

    let logits = GestureClassifier::predict(&imu_sample, &mut arena).unwrap();
}
```

---

## Running the Example

```bash
cargo run --package embedded-nn --example gesture_recognition --features="libm"
```

Or from within this directory:

```bash
cd examples/imu-gesture
cargo run
```

---

## Memory & Allocation

- **Dynamic Heap Allocation (`alloc`)**: **0 bytes**
- **SRAM Buffer (`ARENA_SIZE`)**: **20 bytes**
- **Flash ROM Storage**: ~100 bytes
