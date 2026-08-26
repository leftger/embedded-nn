# Standalone C99 Bare-Metal Neural Network Inference

Demonstrates how to export, compile, and run neural networks in pure **C99** with **zero external runtime dependencies** and **zero dynamic memory allocations (`malloc`)** using `embedded-nn`.

---

## Key Highlights

- **Pure C99 Compatibility**: Suitable for Keil MDK, IAR Embedded Workbench, STM32CubeIDE, ESP-IDF, Zephyr RTOS, and bare-metal GCC/Clang toolchains.
- **Self-Contained Header**: All fixed-point quantization math, weights tables, and static execution routines reside in `include/gesture_model.h`.
- **Zero Dynamic Allocations**: Statically sized SRAM arena (`uint8_t g_arena[32]`) guarantees deterministic memory usage without heap fragmentation.

---

## Generating C99 Headers from Models

You can generate C99 headers from any `.tflite` or `.json` model graph using the `enn` CLI:

```bash
# Export C99 project pack
enn codegen --model models/gesture_model.json --name GestureModel --out include/gesture_model.h
```

---

## Building and Running

### Using Make

```bash
cd examples/c99-baremetal
make run
```

### Using CMake

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
./c99_baremetal_inference
```

### Integrating into Embedded Firmware (STM32 / ESP32 / NXP)

Copy `include/gesture_model.h` into your embedded project's include path and call:

```c
#include "gesture_model.h"

static uint8_t g_arena[GESTUREMODEL_ARENA_SIZE_BYTES];
static int8_t  g_input[GESTUREMODEL_INPUT_DIM];
static int8_t  g_output[GESTUREMODEL_OUTPUT_DIM];

void run_inference(void) {
    // Fill g_input with sensor readings...
    gesturemodel_predict(g_input, g_output, g_arena);
    // Read results from g_output...
}
```
