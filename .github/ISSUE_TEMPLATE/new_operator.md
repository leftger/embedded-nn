---
name: 🧩 New Operator Request
about: Propose or request support for a new neural network operator
title: '[OP] '
labels: ['enhancement', 'operator']
assignees: ''
---

## Operator Name
- **Operator**: (e.g. `GELU`, `LayerNorm`, `PReLU`, `HardSwish`, `BatchMatMul`)
- **TFLite Builtin Op**: (e.g. `GELU`, `HARD_SWISH`)
- **ONNX Op**: (e.g. `Gelu`, `HardSwish`)

## Supported Datatypes
- [ ] Signed 8-bit quantized (`s8` / `int8`)
- [ ] Signed 16-bit quantized (`s16` / `int16`)
- [ ] Floating point (`f32` / `f16`)

## Mathematical Definition
Provide the mathematical formula or reference documentation for this operator.

## Target Use Case
Describe which neural network model or architecture requires this operator (e.g. MobileNetV3, Tiny Transformer, Conformer).
