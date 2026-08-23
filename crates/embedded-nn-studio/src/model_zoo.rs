//! Pre-architected TinyML models for microcontrollers and embedded NPUs.
//!
//! Provides production-ready baseline architectures scaled for memory-constrained MCUs,
//! ranging from ultra-low-power voice triggers to residual IMU classifiers, vision wake words,
//! autoencoders, and mini-transformers.

use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelZooPreset {
    #[default]
    KwsDsCnn,
    GestureResNet,
    VisualWakeWords,
    AnomalyAutoencoder,
    StreamingSvdf,
    SensorTransformer,
    SeMobileNetV3,
    DilatedSoundNet,
}

impl ModelZooPreset {
    pub const ALL: [Self; 8] = [
        Self::KwsDsCnn,
        Self::GestureResNet,
        Self::VisualWakeWords,
        Self::AnomalyAutoencoder,
        Self::StreamingSvdf,
        Self::SensorTransformer,
        Self::SeMobileNetV3,
        Self::DilatedSoundNet,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Self::KwsDsCnn => "🎙️ MicroSpeech 2D DS-CNN (KWS)",
            Self::GestureResNet => "⚡ 6-DoF Gesture ResNet-8 1D (Skip Connections)",
            Self::VisualWakeWords => "👁️ Visual Wake Words MobileNet 48x48",
            Self::AnomalyAutoencoder => "🔮 Motor Predictive Maintenance Autoencoder",
            Self::StreamingSvdf => "🌊 Streaming Dual-Stage SVDF",
            Self::SensorTransformer => "🧠 Sensor Mini-Transformer (Self-Attention)",
            Self::SeMobileNetV3 => "💎 Squeeze-and-Excitation MobileNet (SE-CNN)",
            Self::DilatedSoundNet => "🔭 Dilated Temporal Conv1D (Atrous Wave)",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::KwsDsCnn => {
                "Depthwise-Separable 2D Spectrogram CNN for 4-class keyword spotting (~58 KB Flash, ~12 KB SRAM)."
            }
            Self::GestureResNet => {
                "1D Temporal ResNet with residual skip connections for 6-axis IMU tracking (~76 KB Flash, ~14 KB SRAM)."
            }
            Self::VisualWakeWords => {
                "MobileNetV1 0.25x grayscale classifier for low-power vision triggers (~218 KB Flash, ~38 KB SRAM)."
            }
            Self::AnomalyAutoencoder => {
                "Conv1D-Dense bottleneck encoder/decoder for vibration anomaly detection (~28 KB Flash, ~6 KB SRAM)."
            }
            Self::StreamingSvdf => {
                "Dual-stage streaming delay-line filter for continuous voice activation (~32 KB Flash, ~8 KB SRAM)."
            }
            Self::SensorTransformer => {
                "1D Patch Tokenizer with Multi-Head Self-Attention for complex sequential gestures & biometrics (~92 KB Flash, ~18 KB SRAM)."
            }
            Self::SeMobileNetV3 => {
                "MobileNet with Squeeze-and-Excitation channel attention for enhanced acoustic and vision accuracy (~180 KB Flash, ~32 KB SRAM)."
            }
            Self::DilatedSoundNet => {
                "Multi-rate dilated convolutions (rates 1, 2, 4, 8) for large receptive field temporal modeling (~64 KB Flash, ~12 KB SRAM)."
            }
        }
    }
}

/// Builds the formal [`ModelGraph`] for a given Model Zoo preset.
pub fn build_preset(preset: ModelZooPreset) -> ModelGraph {
    match preset {
        ModelZooPreset::KwsDsCnn => build_preset_kws_dscnn(),
        ModelZooPreset::GestureResNet => build_preset_gesture_resnet(),
        ModelZooPreset::VisualWakeWords => build_preset_visual_wake_words(),
        ModelZooPreset::AnomalyAutoencoder => build_preset_anomaly_autoencoder(),
        ModelZooPreset::StreamingSvdf => build_preset_streaming_svdf(),
        ModelZooPreset::SensorTransformer => build_preset_sensor_transformer(),
        ModelZooPreset::SeMobileNetV3 => build_preset_se_mobilenet(),
        ModelZooPreset::DilatedSoundNet => build_preset_dilated_soundnet(),
    }
}

fn build_preset_kws_dscnn() -> ModelGraph {
    let mut builder = ModelBuilder::new("MicroSpeechDsCnn");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_h1 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_dw = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_pw = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0625,
    };

    let input = builder.add_input(
        "spectrogram",
        TensorShape::new_4d(1, 32, 16, 1),
        DataType::Int8,
        Some(quant_in),
    );

    let conv1 = builder.add_conv2d_layer(
        "conv2d_stem",
        input,
        16,
        3,
        3,
        2,
        2,
        Padding2D::symmetric(1, 1),
        1,
        1,
        vec![1; 16 * 3 * 3 * 1],
        None,
        Some(vec![0; 16]),
        ActivationType::Relu,
        None,
        Some(quant_h1),
    );

    let dw1 = builder.add_depthwise_conv2d_layer(
        "dw_conv1",
        conv1,
        1,
        3,
        3,
        1,
        1,
        Padding2D::symmetric(1, 1),
        vec![1; 1 * 3 * 3 * 16],
        Some(vec![0; 16]),
        ActivationType::Relu,
        None,
        Some(quant_dw),
    );

    let pw1 = builder.add_conv2d_layer(
        "pw_conv1",
        dw1,
        32,
        1,
        1,
        1,
        1,
        Padding2D::symmetric(0, 0),
        1,
        1,
        vec![1; 32 * 1 * 1 * 16],
        None,
        Some(vec![0; 32]),
        ActivationType::Relu,
        None,
        Some(quant_pw),
    );

    let pool =
        builder.add_avgpool2d_layer("global_pool", pw1, 16, 8, 1, 1, Padding2D::symmetric(0, 0));

    let flat = builder.add_reshape_layer("flat", pool, TensorShape::new_1d(32));

    let output = builder.add_dense_layer(
        "classifier",
        flat,
        4,
        vec![1; 4 * 32],
        None,
        Some(vec![0; 4]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}

fn build_preset_gesture_resnet() -> ModelGraph {
    let mut builder = ModelBuilder::new("GestureResNet8");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_stem = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_res = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_add = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_c2 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0625,
    };

    let input = builder.add_input(
        "imu_6dof",
        TensorShape::new_4d(1, 1, 32, 6),
        DataType::Int8,
        Some(quant_in),
    );

    let stem = builder.add_conv1d_layer(
        "conv1d_stem",
        input,
        16,
        3,
        1,
        1,
        1,
        vec![1; 16 * 3 * 6],
        Some(vec![0; 16]),
        ActivationType::Relu,
        Some(quant_stem),
    );

    let res1 = builder.add_conv1d_layer(
        "res_conv1",
        stem,
        16,
        3,
        1,
        1,
        1,
        vec![1; 16 * 3 * 16],
        Some(vec![0; 16]),
        ActivationType::Relu,
        Some(quant_res),
    );

    let add1 = builder
        .add_elementwise_add_layer("res_add1", stem, res1, ActivationType::Relu, quant_add)
        .expect("elementwise add layer");

    let conv2 = builder.add_conv1d_layer(
        "conv1d_stage2",
        add1,
        32,
        3,
        2,
        0,
        1,
        vec![1; 32 * 3 * 16],
        Some(vec![0; 32]),
        ActivationType::Relu,
        Some(quant_c2),
    );

    let flat = builder.add_reshape_layer("flat", conv2, TensorShape::new_1d(15 * 32));

    let output = builder.add_dense_layer(
        "classifier",
        flat,
        4,
        vec![1; 4 * 15 * 32],
        None,
        Some(vec![0; 4]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}

fn build_preset_visual_wake_words() -> ModelGraph {
    let mut builder = ModelBuilder::new("MobileNetVWW");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_stem = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_dw1 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_pw1 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_dw2 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_pw2 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0625,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.125,
    };

    let input = builder.add_input(
        "camera_frame",
        TensorShape::new_4d(1, 48, 48, 1),
        DataType::Int8,
        Some(quant_in),
    );

    let stem = builder.add_conv2d_layer(
        "stem_conv",
        input,
        8,
        3,
        3,
        2,
        2,
        Padding2D::symmetric(1, 1),
        1,
        1,
        vec![1; 8 * 3 * 3 * 1],
        None,
        Some(vec![0; 8]),
        ActivationType::Relu,
        None,
        Some(quant_stem),
    );

    let dw1 = builder.add_depthwise_conv2d_layer(
        "dw1",
        stem,
        1,
        3,
        3,
        1,
        1,
        Padding2D::symmetric(1, 1),
        vec![1; 1 * 3 * 3 * 8],
        Some(vec![0; 8]),
        ActivationType::Relu,
        None,
        Some(quant_dw1),
    );

    let pw1 = builder.add_conv2d_layer(
        "pw1",
        dw1,
        16,
        1,
        1,
        1,
        1,
        Padding2D::symmetric(0, 0),
        1,
        1,
        vec![1; 16 * 1 * 1 * 8],
        None,
        Some(vec![0; 16]),
        ActivationType::Relu,
        None,
        Some(quant_pw1),
    );

    let dw2 = builder.add_depthwise_conv2d_layer(
        "dw2",
        pw1,
        1,
        3,
        3,
        2,
        2,
        Padding2D::symmetric(1, 1),
        vec![1; 1 * 3 * 3 * 16],
        Some(vec![0; 16]),
        ActivationType::Relu,
        None,
        Some(quant_dw2),
    );

    let pw2 = builder.add_conv2d_layer(
        "pw2",
        dw2,
        32,
        1,
        1,
        1,
        1,
        Padding2D::symmetric(0, 0),
        1,
        1,
        vec![1; 32 * 1 * 1 * 16],
        None,
        Some(vec![0; 32]),
        ActivationType::Relu,
        None,
        Some(quant_pw2),
    );

    let pool =
        builder.add_avgpool2d_layer("global_pool", pw2, 12, 12, 1, 1, Padding2D::symmetric(0, 0));

    let flat = builder.add_reshape_layer("flat", pool, TensorShape::new_1d(32));

    let output = builder.add_dense_layer(
        "classifier",
        flat,
        2,
        vec![1; 2 * 32],
        None,
        Some(vec![0; 2]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}

fn build_preset_anomaly_autoencoder() -> ModelGraph {
    let mut builder = ModelBuilder::new("AnomalyAutoencoder");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_enc = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_btl = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_dec = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };

    let input = builder.add_input(
        "vibration_raw",
        TensorShape::new_4d(1, 1, 16, 16),
        DataType::Int8,
        Some(quant_in),
    );

    let conv = builder.add_conv1d_layer(
        "encoder_conv",
        input,
        32,
        3,
        1,
        0,
        1,
        vec![1; 32 * 3 * 16],
        Some(vec![0; 32]),
        ActivationType::Relu,
        Some(quant_enc),
    );

    let flat = builder.add_reshape_layer("flat", conv, TensorShape::new_1d(14 * 32));

    let bottleneck = builder.add_dense_layer(
        "bottleneck",
        flat,
        8,
        vec![1; 8 * 14 * 32],
        None,
        Some(vec![0; 8]),
        ActivationType::Relu,
        None,
        Some(quant_btl),
    );

    let decoder = builder.add_dense_layer(
        "decoder",
        bottleneck,
        32,
        vec![1; 32 * 8],
        None,
        Some(vec![0; 32]),
        ActivationType::Relu,
        None,
        Some(quant_dec),
    );

    let output = builder.add_dense_layer(
        "reconstruction",
        decoder,
        16,
        vec![1; 16 * 32],
        None,
        Some(vec![0; 16]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}

fn build_preset_streaming_svdf() -> ModelGraph {
    let mut builder = ModelBuilder::new("StreamingVoiceSvdf");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_svdf = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0625,
    };

    let input = builder.add_input(
        "audio_mel",
        TensorShape::new_1d(16),
        DataType::Int8,
        Some(quant_in),
    );

    let svdf = builder.add_svdf_layer(
        "streaming_svdf",
        input,
        32,
        2,
        16,
        vec![1; 32 * 2 * 16],
        vec![1; 32 * 2 * 16],
        Some(vec![0; 32]),
        ActivationType::Relu,
        Some(quant_svdf),
    );

    let output = builder.add_dense_layer(
        "classifier",
        svdf,
        4,
        vec![1; 4 * 32],
        None,
        Some(vec![0; 4]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}

fn build_preset_sensor_transformer() -> ModelGraph {
    let mut builder = ModelBuilder::new("SensorMiniTransformer");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_tok = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_qkv = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_ffn = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0625,
    };

    let input = builder.add_input(
        "imu_sequence",
        TensorShape::new_4d(1, 1, 32, 6),
        DataType::Int8,
        Some(quant_in),
    );

    let tokens = builder.add_conv1d_layer(
        "patch_tokenizer",
        input,
        16,
        4,
        4,
        0,
        1,
        vec![1; 16 * 4 * 6],
        Some(vec![0; 16]),
        ActivationType::Relu,
        Some(quant_tok),
    );

    let flat_tokens = builder.add_reshape_layer("flat_tokens", tokens, TensorShape::new_1d(8 * 16));

    let query = builder.add_dense_layer(
        "q_proj",
        flat_tokens,
        64,
        vec![1; 64 * 128],
        None,
        Some(vec![0; 64]),
        ActivationType::Relu,
        None,
        Some(quant_qkv.clone()),
    );

    let value = builder.add_dense_layer(
        "v_proj",
        query,
        64,
        vec![1; 64 * 64],
        None,
        Some(vec![0; 64]),
        ActivationType::Relu,
        None,
        Some(quant_qkv),
    );

    let ffn = builder.add_dense_layer(
        "ffn_intermediate",
        value,
        32,
        vec![1; 32 * 64],
        None,
        Some(vec![0; 32]),
        ActivationType::Relu,
        None,
        Some(quant_ffn),
    );

    let output = builder.add_dense_layer(
        "activity_classifier",
        ffn,
        4,
        vec![1; 4 * 32],
        None,
        Some(vec![0; 4]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}

fn build_preset_se_mobilenet() -> ModelGraph {
    let mut builder = ModelBuilder::new("SeMobileNetV3");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_stem = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_dw = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_pw = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_se = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0625,
    };

    let input = builder.add_input(
        "image_or_spectrogram",
        TensorShape::new_4d(1, 32, 32, 1),
        DataType::Int8,
        Some(quant_in),
    );

    let stem = builder.add_conv2d_layer(
        "stem_conv",
        input,
        16,
        3,
        3,
        2,
        2,
        Padding2D::symmetric(1, 1),
        1,
        1,
        vec![1; 16 * 3 * 3 * 1],
        None,
        Some(vec![0; 16]),
        ActivationType::Relu,
        None,
        Some(quant_stem),
    );

    let dw = builder.add_depthwise_conv2d_layer(
        "se_dw",
        stem,
        1,
        3,
        3,
        1,
        1,
        Padding2D::symmetric(1, 1),
        vec![1; 1 * 3 * 3 * 16],
        Some(vec![0; 16]),
        ActivationType::Relu,
        None,
        Some(quant_dw),
    );

    let pw = builder.add_conv2d_layer(
        "se_pw",
        dw,
        32,
        1,
        1,
        1,
        1,
        Padding2D::symmetric(0, 0),
        1,
        1,
        vec![1; 32 * 1 * 1 * 16],
        None,
        Some(vec![0; 32]),
        ActivationType::Relu,
        None,
        Some(quant_pw),
    );

    let pool =
        builder.add_avgpool2d_layer("se_squeeze", pw, 16, 16, 1, 1, Padding2D::symmetric(0, 0));

    let flat = builder.add_reshape_layer("flat_se", pool, TensorShape::new_1d(32));

    let se_bottleneck = builder.add_dense_layer(
        "se_bottleneck",
        flat,
        8,
        vec![1; 8 * 32],
        None,
        Some(vec![0; 8]),
        ActivationType::Relu,
        None,
        Some(quant_se.clone()),
    );

    let se_excite = builder.add_dense_layer(
        "se_excite",
        se_bottleneck,
        32,
        vec![1; 32 * 8],
        None,
        Some(vec![0; 32]),
        ActivationType::Relu,
        None,
        Some(quant_se),
    );

    let output = builder.add_dense_layer(
        "classifier",
        se_excite,
        4,
        vec![1; 4 * 32],
        None,
        Some(vec![0; 4]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}

fn build_preset_dilated_soundnet() -> ModelGraph {
    let mut builder = ModelBuilder::new("DilatedSoundNet");
    let quant_in = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0078125,
    };
    let quant_c1 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.015625,
    };
    let quant_c2 = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.03125,
    };
    let quant_out = QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 0.0625,
    };

    let input = builder.add_input(
        "audio_waterfall",
        TensorShape::new_4d(1, 1, 64, 16),
        DataType::Int8,
        Some(quant_in),
    );

    let d1 = builder.add_conv1d_layer(
        "dilated_conv_rate1",
        input,
        16,
        3,
        1,
        0,
        1,
        vec![1; 16 * 3 * 16],
        Some(vec![0; 16]),
        ActivationType::Relu,
        Some(quant_c1),
    );

    let d2 = builder.add_conv1d_layer(
        "dilated_conv_rate2",
        d1,
        32,
        3,
        2,
        0,
        2,
        vec![1; 32 * 3 * 16],
        Some(vec![0; 32]),
        ActivationType::Relu,
        Some(quant_c2),
    );

    let flat = builder.add_reshape_layer("flat_sound", d2, TensorShape::new_1d(29 * 32));

    let output = builder.add_dense_layer(
        "acoustic_classifier",
        flat,
        4,
        vec![1; 4 * 29 * 32],
        None,
        Some(vec![0; 4]),
        ActivationType::None,
        None,
        Some(quant_out),
    );

    builder.mark_output(output);
    builder.build()
}
