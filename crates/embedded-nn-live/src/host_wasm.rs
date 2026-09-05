use crate::{DecodeError, EncodeError, Msg};

pub const AGENT_VID: u16 = 0x1209;
pub const AGENT_PID: u16 = 0xE612;
pub const USB_MPS: usize = 512;
pub const DEFAULT_MAX_PAYLOAD_SIZE: usize = 16 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("USB device enumeration failed: {0}")]
    Enumeration(String),
    #[error("USB operation failed: {0}")]
    Usb(String),
    #[error("USB device is no longer connected")]
    DeviceNotFound,
    #[error("no interface has the requested bulk IN/OUT endpoints")]
    EndpointsNotFound,
    #[error("invalid transport configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("payload length {actual} exceeds maximum {max}")]
    Oversized { actual: usize, max: usize },
    #[error("transport timed out")]
    Timeout,
    #[error("transport peer disconnected")]
    Disconnected,
    #[error("transport I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode failed: {0:?}")]
    Encode(EncodeError),
    #[error("decode failed: {0:?}")]
    Decode(DecodeError),
    #[error("protocol handshake failed: {0}")]
    Handshake(&'static str),
    #[error("device nack seq={seq} code={code}")]
    Nack { seq: u32, code: u16 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsbBridge {
    pub vendor_id: u16,
    pub product_id: u16,
    pub serial_number: Option<String>,
    pub product_string: Option<String>,
    pub manufacturer_string: Option<String>,
    pub bus_id: Option<String>,
    pub device_address: Option<u8>,
    pub port_chain: Vec<u8>,
    legacy_product_selector: Option<String>,
}

impl UsbBridge {
    pub fn new(device_name: impl Into<String>) -> Self {
        let device_name = device_name.into();
        Self {
            vendor_id: 0,
            product_id: 0,
            serial_number: None,
            product_string: Some(device_name.clone()),
            manufacturer_string: None,
            bus_id: None,
            device_address: None,
            port_chain: Vec::new(),
            legacy_product_selector: Some(device_name),
        }
    }

    pub fn enumerate_devices() -> Result<Vec<Self>, TransportError> {
        Ok(Vec::new())
    }

    pub fn enumerate_nn_agents() -> Result<Vec<Self>, TransportError> {
        Ok(Vec::new())
    }

    pub fn list_devices() -> Vec<String> {
        Vec::new()
    }

    pub fn format_selector(&self) -> String {
        self.product_string
            .clone()
            .unwrap_or_else(|| "Unknown USB Device".into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OwnedMsg {
    Hello {
        proto: u16,
        model_id: u32,
        input_len: u32,
        output_len: u32,
    },
    RunInference {
        seq: u32,
        model_id: u32,
        input: Vec<u8>,
    },
    Ping,
    Ready {
        proto: u16,
        model_id: u32,
        input_len: u32,
        output_len: u32,
        max_payload: u32,
        cpu_hz: u32,
    },
    InferenceResult {
        seq: u32,
        model_id: u32,
        execution_cycles: u32,
        execution_time_us: u32,
        logits: Vec<u8>,
    },
    SensorFrame {
        timestamp_ms: u32,
        channel_count: u8,
        values: Vec<u8>,
    },
    Nack {
        seq: u32,
        code: u16,
    },
    Pong,
    LayerProfile {
        seq: u32,
        layer_idx: u8,
        total_layers: u8,
        execution_cycles: u32,
        activations: Vec<u8>,
    },
}

impl OwnedMsg {
    pub fn from_msg(msg: Msg<'_>) -> Self {
        match msg {
            Msg::Hello {
                proto,
                model_id,
                input_len,
                output_len,
            } => Self::Hello {
                proto,
                model_id,
                input_len,
                output_len,
            },
            Msg::RunInference {
                seq,
                model_id,
                input,
            } => Self::RunInference {
                seq,
                model_id,
                input: input.to_vec(),
            },
            Msg::Ping => Self::Ping,
            Msg::Ready {
                proto,
                model_id,
                input_len,
                output_len,
                max_payload,
                cpu_hz,
            } => Self::Ready {
                proto,
                model_id,
                input_len,
                output_len,
                max_payload,
                cpu_hz,
            },
            Msg::InferenceResult {
                seq,
                model_id,
                execution_cycles,
                execution_time_us,
                logits,
            } => Self::InferenceResult {
                seq,
                model_id,
                execution_cycles,
                execution_time_us,
                logits: logits.to_vec(),
            },
            Msg::SensorFrame {
                timestamp_ms,
                channel_count,
                values,
            } => Self::SensorFrame {
                timestamp_ms,
                channel_count,
                values: values.to_vec(),
            },
            Msg::Nack { seq, code } => Self::Nack {
                seq,
                code: code as u16,
            },
            Msg::Pong => Self::Pong,
            Msg::LayerProfile {
                seq,
                layer_idx,
                total_layers,
                execution_cycles,
                activations,
            } => Self::LayerProfile {
                seq,
                layer_idx,
                total_layers,
                execution_cycles,
                activations: activations.to_vec(),
            },
        }
    }

    pub fn to_msg(&self) -> Msg<'_> {
        match self {
            Self::Hello {
                proto,
                model_id,
                input_len,
                output_len,
            } => Msg::Hello {
                proto: *proto,
                model_id: *model_id,
                input_len: *input_len,
                output_len: *output_len,
            },
            Self::RunInference {
                seq,
                model_id,
                input,
            } => Msg::RunInference {
                seq: *seq,
                model_id: *model_id,
                input,
            },
            Self::Ping => Msg::Ping,
            Self::Ready {
                proto,
                model_id,
                input_len,
                output_len,
                max_payload,
                cpu_hz,
            } => Msg::Ready {
                proto: *proto,
                model_id: *model_id,
                input_len: *input_len,
                output_len: *output_len,
                max_payload: *max_payload,
                cpu_hz: *cpu_hz,
            },
            Self::InferenceResult {
                seq,
                model_id,
                execution_cycles,
                execution_time_us,
                logits,
            } => Msg::InferenceResult {
                seq: *seq,
                model_id: *model_id,
                execution_cycles: *execution_cycles,
                execution_time_us: *execution_time_us,
                logits,
            },
            Self::SensorFrame {
                timestamp_ms,
                channel_count,
                values,
            } => Msg::SensorFrame {
                timestamp_ms: *timestamp_ms,
                channel_count: *channel_count,
                values,
            },
            Self::Nack { seq, code } => Msg::Nack {
                seq: *seq,
                code: *code,
            },
            Self::Pong => Msg::Pong,
            Self::LayerProfile {
                seq,
                layer_idx,
                total_layers,
                execution_cycles,
                activations,
            } => Msg::LayerProfile {
                seq: *seq,
                layer_idx: *layer_idx,
                total_layers: *total_layers,
                execution_cycles: *execution_cycles,
                activations,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawFrame {
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceLink {
    device_id: String,
}

impl DeviceLink {
    pub fn connect(device_id: &str) -> Result<Self, String> {
        Err(format!(
            "USB HIL is not supported in WebAssembly environment (device: {device_id})"
        ))
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub fn is_alive(&self) -> bool {
        false
    }

    pub fn is_handshaked(&self) -> bool {
        false
    }

    pub fn ready_info(&self) -> Option<OwnedMsg> {
        None
    }

    pub fn take_error(&self) -> Option<String> {
        None
    }

    pub fn drain_sensors(&self) -> Vec<OwnedMsg> {
        Vec::new()
    }

    pub fn take_sensor(&self) -> Option<OwnedMsg> {
        None
    }

    pub fn take_result(&self) -> Option<OwnedMsg> {
        None
    }

    pub fn ping(&self) {}

    pub fn infer(&self, _seq: u32, _model_id: u32, _input: Vec<u8>) {}
}
