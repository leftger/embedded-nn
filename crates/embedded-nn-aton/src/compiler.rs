//! High-level ATON NPU graph compiler and code generator.

use crate::container::EcContainerBuilder;
use crate::dma::{DmaChannel, ElementSize, StrengConfig};
use crate::error::AtonError;
use crate::isa::*;
use crate::switch::{SwitchDst, SwitchRoute, SwitchSrc, emit_switch_routes};
use embedded_nn_compiler::ir::{
    ActivationType, DataType, ModelGraph, OpPayload, Padding2D, TensorShape,
};

/// Classification of an epoch in the hybrid execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpochKind {
    /// Hardware epoch scheduled to run on the Neural-ART (ATON) NPU.
    Hardware {
        layer_ids: Vec<usize>,
        input_symbol: String,
        output_symbol: String,
    },
    /// Software fallback epoch scheduled to run on the Cortex-M55 CPU.
    Software { layer_id: usize, op_name: String },
}

/// A compiled TinyML network ready for deployment on the STM32N6 ATON NPU.
pub struct AtonCompiledNetwork {
    pub name: String,
    pub container: EcContainerBuilder,
    pub epochs: Vec<EpochKind>,
    pub static_weights: Vec<u8>,
    pub input_size_bytes: usize,
    pub output_size_bytes: usize,
}

impl AtonCompiledNetwork {
    /// Builds the complete `EcBinary` container as aligned 64-bit words for `BlobSession`.
    pub fn build_blob_u64(&self) -> Result<Vec<u64>, AtonError> {
        self.container.build_words_u64()
    }

    /// Emits C header format `<name>_ecblobs.h`.
    pub fn emit_c_header(&self) -> Result<String, AtonError> {
        self.container.emit_c_header(&self.name)
    }

    /// Emits Rust source code defining the aligned static container.
    pub fn emit_rust_code(&self) -> Result<String, AtonError> {
        self.container.emit_rust_code(&self.name)
    }
}

/// Parameters for configuring a Convolutional Accelerator unit (CONVACC).
#[derive(Debug, Clone)]
pub struct ConvaccParams<'a> {
    pub unit: usize,
    pub in_shape: &'a TensorShape,
    pub out_shape: &'a TensorShape,
    pub kernel_w: usize,
    pub kernel_h: usize,
    pub stride_w: usize,
    pub stride_h: usize,
    pub padding: &'a Padding2D,
    pub is_depthwise: bool,
    pub ch_mult: usize,
    pub activation: &'a ActivationType,
}

/// Parameters for configuring a Pooling Accelerator unit (POOLACC).
#[derive(Debug, Clone)]
pub struct PoolaccParams<'a> {
    pub unit: usize,
    pub in_shape: &'a TensorShape,
    pub out_shape: &'a TensorShape,
    pub win_w: usize,
    pub win_h: usize,
    pub stride_w: usize,
    pub stride_h: usize,
    pub padding: &'a Padding2D,
    pub op: u32,
    pub mulval: u16,
    pub activation: &'a ActivationType,
}

/// Parameters for configuring an Arithmetic Accelerator unit (ARITHACC).
#[derive(Debug, Clone)]
pub struct ArithaccParams<'a> {
    pub unit: usize,
    pub op: u32,
    pub dual_input: bool,
    pub coeff_a: i16,
    pub coeff_b: i16,
    pub coeff_c: i16,
    pub ax_shift: u8,
    pub by_shift: u8,
    pub res_shift: u8,
    pub activation: &'a ActivationType,
}

/// Open-source ATON compiler mapping high-level computation graphs to STM32N6 NPU hardware.
pub struct AtonCompiler {
    input_sym: String,
    output_sym: String,
}

impl Default for AtonCompiler {
    fn default() -> Self {
        Self {
            input_sym: "_user_io_input_0".into(),
            output_sym: "_user_io_output_0".into(),
        }
    }
}

impl AtonCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets custom relocation symbol names for input and output.
    pub fn with_symbols(
        mut self,
        input_sym: impl Into<String>,
        output_sym: impl Into<String>,
    ) -> Self {
        self.input_sym = input_sym.into();
        self.output_sym = output_sym.into();
        self
    }

    /// Compiles a `ModelGraph` into an `AtonCompiledNetwork`.
    pub fn compile(&self, graph: &ModelGraph) -> Result<AtonCompiledNetwork, AtonError> {
        let mut container = EcContainerBuilder::new();
        let mut epochs = Vec::new();
        let mut static_weights = Vec::new();

        // 1. Partition graph into hardware and software epochs
        let mut hw_layers = Vec::new();
        for layer in &graph.layers {
            let is_hw = match &layer.op {
                OpPayload::FullyConnected { .. }
                | OpPayload::Conv2D { .. }
                | OpPayload::DepthwiseConv2D { .. }
                | OpPayload::MaxPool2D { .. }
                | OpPayload::AvgPool2D { .. }
                | OpPayload::ElementwiseAdd { .. }
                | OpPayload::ElementwiseMul { .. } => true,
                OpPayload::Mean {
                    reduce_height,
                    reduce_width,
                    ..
                } => *reduce_height && *reduce_width,
                _ => false,
            };

            if is_hw {
                hw_layers.push(layer.id);
            } else {
                // Non-NPU layer (Softmax, GELU, custom activation) runs as software fallback epoch
                if !hw_layers.is_empty() {
                    epochs.push(EpochKind::Hardware {
                        layer_ids: hw_layers.clone(),
                        input_symbol: self.input_sym.clone(),
                        output_symbol: format!("_intermediate_{}", layer.id),
                    });
                    hw_layers.clear();
                }
                epochs.push(EpochKind::Software {
                    layer_id: layer.id,
                    op_name: layer.name.clone(),
                });
            }
        }

        if !hw_layers.is_empty() {
            epochs.push(EpochKind::Hardware {
                layer_ids: hw_layers.clone(),
                input_symbol: self.input_sym.clone(),
                output_symbol: self.output_sym.clone(),
            });
        }

        // 2. Generate ATON microcode instructions
        let mut instructions = Vec::new();

        // Step A: Clock Gating & Unit Initialization
        instructions.push(EcInstruction::WriteReg {
            reg_addr: CLKCTRL_CTRL,
            value: 1, // enable clock controller
        });
        instructions.push(EcInstruction::WriteReg {
            reg_addr: CLKCTRL_AGATES0,
            value: 0xFFFF_FFFF, // ungate all infrastructure clocks
        });
        instructions.push(EcInstruction::WriteReg {
            reg_addr: CLKCTRL_BGATES,
            value: 0x07FF_FFFF, // ungate accelerator execution units
        });

        // Step B: Collect primary input/output tensor shapes
        let input_tensor_id = *graph
            .inputs
            .first()
            .ok_or_else(|| AtonError::InvalidShape("Graph has no inputs".into()))?;
        let input_tensor = graph
            .tensors
            .iter()
            .find(|t| t.id == input_tensor_id)
            .ok_or(AtonError::TensorNotFound(input_tensor_id))?;

        let output_tensor_id = *graph
            .outputs
            .first()
            .ok_or_else(|| AtonError::InvalidShape("Graph has no outputs".into()))?;
        let output_tensor = graph
            .tensors
            .iter()
            .find(|t| t.id == output_tensor_id)
            .ok_or(AtonError::TensorNotFound(output_tensor_id))?;

        let input_size = input_tensor.shape.total_elements();
        let output_size = output_tensor.shape.total_elements();

        // Step C: Iterate over hardware layers, configure execution units, and build switch routes
        let mut routes = Vec::new();
        let mut current_src = SwitchSrc::Streng(0);
        let mut has_dual_input = false;
        let mut dual_input_shape = None;
        let mut dual_input_dtype = DataType::Int8;

        for &layer_id in &hw_layers {
            let layer = graph
                .layers
                .iter()
                .find(|l| l.id == layer_id)
                .ok_or_else(|| {
                    AtonError::UnsupportedLayer(format!("Layer {} not found", layer_id))
                })?;

            let in_t = graph
                .tensors
                .iter()
                .find(|t| t.id == layer.inputs[0])
                .ok_or(AtonError::TensorNotFound(layer.inputs[0]))?;

            let out_t = graph
                .tensors
                .iter()
                .find(|t| t.id == layer.outputs[0])
                .ok_or(AtonError::TensorNotFound(layer.outputs[0]))?;

            match &layer.op {
                OpPayload::FullyConnected { activation, .. } => {
                    let fc_in = TensorShape::new_4d(1, 1, 1, in_t.shape.total_elements());
                    let fc_out = TensorShape::new_4d(1, 1, 1, out_t.shape.total_elements());
                    emit_convacc_config(
                        &ConvaccParams {
                            unit: 0,
                            in_shape: &fc_in,
                            out_shape: &fc_out,
                            kernel_w: 1,
                            kernel_h: 1,
                            stride_w: 1,
                            stride_h: 1,
                            padding: &Padding2D::default(),
                            is_depthwise: false,
                            ch_mult: 1,
                            activation,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: current_src,
                        dst: SwitchDst::ConvInA,
                    });
                    if matches!(activation, ActivationType::Relu6) {
                        emit_activacc_config(0, ACTIV_OP_TRELU, 6, 0, &mut instructions);
                        routes.push(SwitchRoute {
                            src: SwitchSrc::ConvOut,
                            dst: SwitchDst::ActUnitIn,
                        });
                        current_src = SwitchSrc::ActUnitOut;
                    } else {
                        current_src = SwitchSrc::ConvOut;
                    }
                }

                OpPayload::Conv2D {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding,
                    activation,
                    ..
                } => {
                    emit_convacc_config(
                        &ConvaccParams {
                            unit: 0,
                            in_shape: &in_t.shape,
                            out_shape: &out_t.shape,
                            kernel_w: *kernel_w,
                            kernel_h: *kernel_h,
                            stride_w: *stride_w,
                            stride_h: *stride_h,
                            padding,
                            is_depthwise: false,
                            ch_mult: 1,
                            activation,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: current_src,
                        dst: SwitchDst::ConvInA,
                    });
                    if matches!(activation, ActivationType::Relu6) {
                        emit_activacc_config(0, ACTIV_OP_TRELU, 6, 0, &mut instructions);
                        routes.push(SwitchRoute {
                            src: SwitchSrc::ConvOut,
                            dst: SwitchDst::ActUnitIn,
                        });
                        current_src = SwitchSrc::ActUnitOut;
                    } else {
                        current_src = SwitchSrc::ConvOut;
                    }
                }

                OpPayload::DepthwiseConv2D {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding,
                    ch_mult,
                    activation,
                    ..
                } => {
                    emit_convacc_config(
                        &ConvaccParams {
                            unit: 0,
                            in_shape: &in_t.shape,
                            out_shape: &out_t.shape,
                            kernel_w: *kernel_w,
                            kernel_h: *kernel_h,
                            stride_w: *stride_w,
                            stride_h: *stride_h,
                            padding,
                            is_depthwise: true,
                            ch_mult: *ch_mult,
                            activation,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: current_src,
                        dst: SwitchDst::ConvInA,
                    });
                    if matches!(activation, ActivationType::Relu6) {
                        emit_activacc_config(0, ACTIV_OP_TRELU, 6, 0, &mut instructions);
                        routes.push(SwitchRoute {
                            src: SwitchSrc::ConvOut,
                            dst: SwitchDst::ActUnitIn,
                        });
                        current_src = SwitchSrc::ActUnitOut;
                    } else {
                        current_src = SwitchSrc::ConvOut;
                    }
                }

                OpPayload::MaxPool2D {
                    pool_h,
                    pool_w,
                    stride_h,
                    stride_w,
                    padding,
                } => {
                    emit_poolacc_config(
                        &PoolaccParams {
                            unit: 0,
                            in_shape: &in_t.shape,
                            out_shape: &out_t.shape,
                            win_w: *pool_w,
                            win_h: *pool_h,
                            stride_w: *stride_w,
                            stride_h: *stride_h,
                            padding,
                            op: POOL_OP_MAX,
                            mulval: 0,
                            activation: &ActivationType::None,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: current_src,
                        dst: SwitchDst::PoolIn,
                    });
                    current_src = SwitchSrc::PoolOut;
                }

                OpPayload::AvgPool2D {
                    pool_h,
                    pool_w,
                    stride_h,
                    stride_w,
                    padding,
                } => {
                    let area = (pool_h * pool_w).max(1);
                    let mulval = ((1u32 << 16) / area as u32) as u16;
                    emit_poolacc_config(
                        &PoolaccParams {
                            unit: 0,
                            in_shape: &in_t.shape,
                            out_shape: &out_t.shape,
                            win_w: *pool_w,
                            win_h: *pool_h,
                            stride_w: *stride_w,
                            stride_h: *stride_h,
                            padding,
                            op: POOL_OP_AVG,
                            mulval,
                            activation: &ActivationType::None,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: current_src,
                        dst: SwitchDst::PoolIn,
                    });
                    current_src = SwitchSrc::PoolOut;
                }

                OpPayload::Mean {
                    reduce_height,
                    reduce_width,
                    ..
                } if *reduce_height && *reduce_width => {
                    let area = (in_t.shape.height * in_t.shape.width).max(1);
                    let mulval = ((1u32 << 16) / area as u32) as u16;
                    let mean_out = TensorShape::new_4d(1, 1, 1, in_t.shape.channels);
                    emit_poolacc_config(
                        &PoolaccParams {
                            unit: 0,
                            in_shape: &in_t.shape,
                            out_shape: &mean_out,
                            win_w: in_t.shape.width,
                            win_h: in_t.shape.height,
                            stride_w: 1,
                            stride_h: 1,
                            padding: &Padding2D::default(),
                            op: POOL_OP_GAVG,
                            mulval,
                            activation: &ActivationType::None,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: current_src,
                        dst: SwitchDst::PoolIn,
                    });
                    current_src = SwitchSrc::PoolOut;
                }

                OpPayload::ElementwiseAdd { quant, activation } => {
                    has_dual_input = true;
                    if layer.inputs.len() > 1 {
                        let in2_t = graph
                            .tensors
                            .iter()
                            .find(|t| t.id == layer.inputs[1])
                            .ok_or(AtonError::TensorNotFound(layer.inputs[1]))?;
                        dual_input_shape = Some(in2_t.shape);
                        dual_input_dtype = in2_t.dtype;
                    } else {
                        dual_input_shape = Some(in_t.shape);
                        dual_input_dtype = in_t.dtype;
                    }

                    emit_arithacc_config(
                        &ArithaccParams {
                            unit: 0,
                            op: ARITH_OP_AFFINE,
                            dual_input: true,
                            coeff_a: (quant.input1_multiplier >> 16) as i16,
                            coeff_b: (quant.input2_multiplier >> 16) as i16,
                            coeff_c: quant.output_offset as i16,
                            ax_shift: quant.input1_shift.max(0) as u8,
                            by_shift: quant.input2_shift.max(0) as u8,
                            res_shift: quant.output_shift.max(0) as u8,
                            activation,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: SwitchSrc::Streng(0),
                        dst: SwitchDst::ArithInX,
                    });
                    routes.push(SwitchRoute {
                        src: SwitchSrc::Streng(1),
                        dst: SwitchDst::ArithInY,
                    });
                    current_src = SwitchSrc::ArithOut;
                }

                OpPayload::ElementwiseMul { quant, activation } => {
                    has_dual_input = true;
                    if layer.inputs.len() > 1 {
                        let in2_t = graph
                            .tensors
                            .iter()
                            .find(|t| t.id == layer.inputs[1])
                            .ok_or(AtonError::TensorNotFound(layer.inputs[1]))?;
                        dual_input_shape = Some(in2_t.shape);
                        dual_input_dtype = in2_t.dtype;
                    } else {
                        dual_input_shape = Some(in_t.shape);
                        dual_input_dtype = in_t.dtype;
                    }

                    emit_arithacc_config(
                        &ArithaccParams {
                            unit: 0,
                            op: ARITH_OP_MUL,
                            dual_input: true,
                            coeff_a: 1,
                            coeff_b: 1,
                            coeff_c: 0,
                            ax_shift: 0,
                            by_shift: 0,
                            res_shift: quant.output_shift.max(0) as u8,
                            activation,
                        },
                        &mut instructions,
                    );
                    routes.push(SwitchRoute {
                        src: SwitchSrc::Streng(0),
                        dst: SwitchDst::ArithInX,
                    });
                    routes.push(SwitchRoute {
                        src: SwitchSrc::Streng(1),
                        dst: SwitchDst::ArithInY,
                    });
                    current_src = SwitchSrc::ArithOut;
                }

                _ => {}
            }
        }

        // Connect the final accelerator output to STRENG 3 for writeback
        routes.push(SwitchRoute {
            src: current_src,
            dst: SwitchDst::Streng(3),
        });

        // Step D: Setup DMA Streaming Engines
        // STRENG 0: Primary Input Tensor
        let in_dma = StrengConfig::new_tensor_shape(
            DmaChannel::InputTensor,
            0x0000_0000,
            &input_tensor.shape,
            map_element_size(input_tensor.dtype),
        );
        in_dma.emit_instructions(&mut instructions);

        // STRENG 1: Secondary input tensor if dual-input operation
        if has_dual_input {
            let shape = dual_input_shape.unwrap_or(input_tensor.shape);
            let in2_dma = StrengConfig::new_tensor_shape(
                DmaChannel::WeightMatrix,
                0x0000_0000,
                &shape,
                map_element_size(dual_input_dtype),
            );
            in2_dma.emit_instructions(&mut instructions);
        }

        // STRENG 3: Output Tensor
        let out_dma = StrengConfig::new_tensor_shape(
            DmaChannel::OutputTensor,
            0x0000_0000,
            &output_tensor.shape,
            map_element_size(output_tensor.dtype),
        );
        out_dma.emit_instructions(&mut instructions);

        // Step E: Emit crossbar routing instructions
        emit_switch_routes(&routes, &mut instructions);

        // Step F: Synchronization barrier & termination
        // Wait for STRENG 3 frame transfer completion (EN_OFLOW_FRM = 1 << 19)
        instructions.push(EcInstruction::WaitEvents {
            event_mask: 1 << 19,
        });

        // Trigger end-of-epoch interrupt
        instructions.push(EcInstruction::TriggerIrq);
        instructions.push(EcInstruction::End);

        // 3. Encode instructions into binary words and calculate relocation offsets
        let mut encoded_words = Vec::new();
        let mut input_reloc_word = None;
        let mut input2_reloc_word = None;
        let mut output_reloc_word = None;

        for instr in &instructions {
            let start_offset = encoded_words.len();
            instr.encode(&mut encoded_words);

            if let EcInstruction::WriteReg { reg_addr, .. } = instr {
                if *reg_addr == streng_addr(0) {
                    input_reloc_word = Some((start_offset + 1) as u32);
                } else if *reg_addr == streng_addr(1) && has_dual_input {
                    input2_reloc_word = Some((start_offset + 1) as u32);
                } else if *reg_addr == streng_addr(3) {
                    output_reloc_word = Some((start_offset + 1) as u32);
                }
            }
        }

        container.extend_instructions(encoded_words);

        // 4. Register relocations in the container
        if let Some(in_off) = input_reloc_word {
            container.add_relocation(&self.input_sym, in_off);
        }
        if let Some(in2_off) = input2_reloc_word {
            let input2_sym = if self.input_sym.ends_with("_0") {
                format!("{}_1", &self.input_sym[..self.input_sym.len() - 2])
            } else {
                format!("{}_1", self.input_sym)
            };
            container.add_relocation(&input2_sym, in2_off);
        }
        if let Some(out_off) = output_reloc_word {
            container.add_relocation(&self.output_sym, out_off);
        }

        // 5. Pack static weights and biases from all layers
        for layer in &graph.layers {
            match &layer.op {
                OpPayload::FullyConnected { weights, bias, .. }
                | OpPayload::Conv2D { weights, bias, .. }
                | OpPayload::DepthwiseConv2D { weights, bias, .. }
                | OpPayload::Conv1D { weights, bias, .. } => {
                    for &w in weights {
                        static_weights.push(w as u8);
                    }
                    if let Some(b) = bias {
                        for &val in b {
                            static_weights.extend_from_slice(&val.to_le_bytes());
                        }
                    }
                }
                OpPayload::LstmStep {
                    input_weights,
                    recurrent_weights,
                    bias,
                    ..
                } => {
                    for &w in input_weights {
                        static_weights.push(w as u8);
                    }
                    for &w in recurrent_weights {
                        static_weights.push(w as u8);
                    }
                    for &val in bias {
                        static_weights.extend_from_slice(&val.to_le_bytes());
                    }
                }
                OpPayload::Gelu { lut, .. } => {
                    for &b in lut {
                        static_weights.push(b as u8);
                    }
                }
                _ => {}
            }
        }

        Ok(AtonCompiledNetwork {
            name: graph.name.clone(),
            container,
            epochs,
            static_weights,
            input_size_bytes: input_size,
            output_size_bytes: output_size,
        })
    }
}

fn map_element_size(dtype: DataType) -> ElementSize {
    match dtype {
        DataType::Int8 => ElementSize::Int8,
        DataType::Int16 => ElementSize::Int16,
        DataType::Float32 => ElementSize::Float32,
        DataType::Int4 => ElementSize::Int8,
    }
}

fn emit_convacc_config(params: &ConvaccParams, out: &mut Vec<EcInstruction>) {
    // 1. CONVACC_CTRL
    // EN (bit 0), NOSUM (bit 8), SIMD (bit 9 = 1 for 8-bit), DSS2MODE (bit 17 if depthwise)
    let mut ctrl = 1u32 | (1 << 8) | (1 << 9);
    if params.is_depthwise && params.stride_w == 2 {
        ctrl |= 1 << 17;
    }
    out.push(EcInstruction::WriteReg {
        reg_addr: convacc_ctrl(params.unit),
        value: ctrl,
    });

    // 2. CONVACC_KFORMAT
    // WIDTH [3:0], HEIGHT [7:4], BTCDEPTH [23:8], NR [31:24]
    let k_depth = if params.is_depthwise {
        1
    } else {
        params.in_shape.channels
    };
    let k_nr = if params.is_depthwise {
        params.in_shape.channels * params.ch_mult
    } else {
        params.out_shape.channels
    };
    let kformat = ((params.kernel_w as u32) & 0xF)
        | (((params.kernel_h as u32) & 0xF) << 4)
        | (((k_depth as u32) & 0xFFFF) << 8)
        | (((k_nr as u32) & 0xFF) << 24);
    out.push(EcInstruction::WriteReg {
        reg_addr: convacc_kformat(params.unit),
        value: kformat,
    });

    // 3. CONVACC_FFORMAT
    // WIDTH [15:0] = in_w * batches, HEIGHT [31:16] = in_h
    let fformat = ((params.in_shape.width.max(1) * params.in_shape.batches.max(1)) as u32 & 0xFFFF)
        | (((params.in_shape.height.max(1)) as u32 & 0xFFFF) << 16);
    out.push(EcInstruction::WriteReg {
        reg_addr: convacc_fformat(params.unit),
        value: fformat,
    });

    // 4. CONVACC_SAMPLE
    // TPAD [2:0], BPAD [5:3], LPAD [8:6], RPAD [11:9], HSTRD [15:12], VSTRD [19:16]
    let sample = (params.padding.top as u32 & 0x7)
        | ((params.padding.bottom as u32 & 0x7) << 3)
        | ((params.padding.left as u32 & 0x7) << 6)
        | ((params.padding.right as u32 & 0x7) << 9)
        | ((params.stride_w.max(1) as u32 & 0xF) << 12)
        | ((params.stride_h.max(1) as u32 & 0xF) << 16);
    out.push(EcInstruction::WriteReg {
        reg_addr: convacc_sample(params.unit),
        value: sample,
    });

    // 5. CONVACC_DFORMAT
    // FBYTES [1:0] = 1, OBYTES [5:4] = 1, FROUND (bit 8), ROUND (bit 10), RELU (bit 11)
    let mut dformat = 1u32 | (1 << 4) | (1 << 8) | (1 << 10);
    if matches!(
        params.activation,
        ActivationType::Relu | ActivationType::Relu6
    ) {
        dformat |= 1 << 11;
    }
    out.push(EcInstruction::WriteReg {
        reg_addr: convacc_dformat(params.unit),
        value: dformat,
    });
}

fn emit_poolacc_config(params: &PoolaccParams, out: &mut Vec<EcInstruction>) {
    // 1. POOLACC_CTRL
    // EN (bit 0), TYPE [5:2] = op, ROUND (bit 6), SAT (bit 7), OUTSHIFT [13:8]
    // FBYTES [17:16] = 1, FROUND (bit 24), FSAT (bit 25)
    let is_avg = params.op == POOL_OP_AVG || params.op == POOL_OP_GAVG;
    let mut ctrl = 1u32 | (params.op << 2) | (1 << 16) | (1 << 24);
    if is_avg {
        ctrl |= (1 << 6) | (1 << 7) | (16 << 8); // ROUND, SAT, OUTSHIFT=16
    }
    out.push(EcInstruction::WriteReg {
        reg_addr: poolacc_ctrl(params.unit),
        value: ctrl,
    });

    // 2. POOLACC_PDIMS
    // WINX [2:0], WINY [5:3], STRDX [9:6], STRDY [13:10], TPAD [16:14], BPAD [19:17], LPAD [22:20], RPAD [25:23], BSIZE [29:26]
    let pdims = (params.win_w.min(7) as u32 & 0x7)
        | ((params.win_h.min(7) as u32 & 0x7) << 3)
        | ((params.stride_w.min(15) as u32 & 0xF) << 6)
        | ((params.stride_h.min(15) as u32 & 0xF) << 10)
        | ((params.padding.top.min(7) as u32 & 0x7) << 14)
        | ((params.padding.bottom.min(7) as u32 & 0x7) << 17)
        | ((params.padding.left.min(7) as u32 & 0x7) << 20)
        | ((params.padding.right.min(7) as u32 & 0x7) << 23)
        | (1 << 26); // BSIZE = 1
    out.push(EcInstruction::WriteReg {
        reg_addr: poolacc_pdims(params.unit),
        value: pdims,
    });

    // 3. POOLACC_FDIMS
    // FEATX [15:0] = in_w * in_c, FEATY [31:16] = in_h
    let fdims = ((params.in_shape.width.max(1) * params.in_shape.channels.max(1)) as u32 & 0xFFFF)
        | ((params.in_shape.height.max(1) as u32 & 0xFFFF) << 16);
    out.push(EcInstruction::WriteReg {
        reg_addr: poolacc_fdims(params.unit),
        value: fdims,
    });

    // 4. POOLACC_OUTDIMS
    // FEATX [15:0] = out_w * out_c, FEATY [31:16] = out_h
    let outdims = ((params.out_shape.width.max(1) * params.out_shape.channels.max(1)) as u32
        & 0xFFFF)
        | ((params.out_shape.height.max(1) as u32 & 0xFFFF) << 16);
    out.push(EcInstruction::WriteReg {
        reg_addr: poolacc_outdims(params.unit),
        value: outdims,
    });

    // 5. POOLACC_MULVAL (if average pooling)
    if is_avg {
        out.push(EcInstruction::WriteReg {
            reg_addr: poolacc_mulval(params.unit),
            value: params.mulval as u32,
        });
    }

    // 6. POOLACC_RNDCTRL
    // FOBYTES [1:0] = 1, OBYTES [5:4] = 1, RELU (bit 7)
    let mut rndctrl = 1u32 | (1 << 4);
    if matches!(
        params.activation,
        ActivationType::Relu | ActivationType::Relu6
    ) {
        rndctrl |= 1 << 7;
    }
    out.push(EcInstruction::WriteReg {
        reg_addr: poolacc_rndctrl(params.unit),
        value: rndctrl,
    });
}

fn emit_arithacc_config(params: &ArithaccParams, out: &mut Vec<EcInstruction>) {
    // 1. ARITHACC_CTRL
    // OP [4:0], ROUND (bit 5), SAT (bit 6), OBYTES [8:7] = 1, DUALIN (bit 11), CLIPOUT (bit 12), RELU (bit 13)
    let is_relu = matches!(
        params.activation,
        ActivationType::Relu | ActivationType::Relu6
    );
    let is_clip = matches!(params.activation, ActivationType::Relu6);
    let mut ctrl = (params.op & 0x1F) | (1 << 5) | (1 << 6) | (1 << 7);
    if params.dual_input {
        ctrl |= 1 << 11;
    }
    if is_clip {
        ctrl |= 1 << 12;
    }
    if is_relu {
        ctrl |= 1 << 13;
    }
    out.push(EcInstruction::WriteReg {
        reg_addr: arithacc_ctrl(params.unit),
        value: ctrl,
    });

    // 2. ARITHACC_SHIFT
    // AX [5:0], BY [11:6], C [17:12], RES [27:22]
    let shift_reg = (params.ax_shift as u32 & 0x3F)
        | ((params.by_shift as u32 & 0x3F) << 6)
        | ((params.res_shift as u32 & 0x3F) << 22);
    out.push(EcInstruction::WriteReg {
        reg_addr: arithacc_shift(params.unit),
        value: shift_reg,
    });

    // 3. ARITHACC_COEFFAC: A in [15:0], C in [31:16]
    let coeff_ac = (params.coeff_a as u16 as u32) | ((params.coeff_c as u16 as u32) << 16);
    out.push(EcInstruction::WriteReg {
        reg_addr: arithacc_coeffac(params.unit),
        value: coeff_ac,
    });

    // 4. ARITHACC_COEFFB: B in [15:0]
    out.push(EcInstruction::WriteReg {
        reg_addr: arithacc_coeffb(params.unit),
        value: params.coeff_b as u16 as u32,
    });

    // 5. ARITHACC_INSHIFTER: FBYTESX [1:0]=1, FBYTESY [17:16]=1
    let inshifter = 1u32 | (1 << 16);
    out.push(EcInstruction::WriteReg {
        reg_addr: arithacc_inshifter(params.unit),
        value: inshifter,
    });

    // 6. ARITHACC_CLIPRANGE (if ReLU6)
    if is_clip {
        let clip = 6u32 << 16; // MIN in [15:0] = 0, MAX in [31:16] = 6
        out.push(EcInstruction::WriteReg {
            reg_addr: arithacc_cliprange(params.unit),
            value: clip,
        });
    }
}

fn emit_activacc_config(
    unit: usize,
    op: u32,
    param: u32,
    param2: u32,
    out: &mut Vec<EcInstruction>,
) {
    // 1. ACTIVACC_CTRL
    // TYPE [3:0], FBYTES [5:4] = 1, OBYTES [9:8] = 1, ROUND (bit 10), SAT (bit 11)
    let ctrl = (op & 0xF) | (1 << 4) | (1 << 8) | (1 << 10) | (1 << 11);
    out.push(EcInstruction::WriteReg {
        reg_addr: activacc_ctrl(unit),
        value: ctrl,
    });

    // 2. ACTIVACC_ACTIVPARAM
    out.push(EcInstruction::WriteReg {
        reg_addr: activacc_activparam(unit),
        value: param,
    });

    // 3. ACTIVACC_ACTIVPARAM2
    out.push(EcInstruction::WriteReg {
        reg_addr: activacc_activparam2(unit),
        value: param2,
    });

    // 4. ACTIVACC_FUNC
    out.push(EcInstruction::WriteReg {
        reg_addr: activacc_func(unit),
        value: 1, // signedop = 1
    });
}
