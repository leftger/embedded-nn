use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorFrame {
    pub timestamp_ms: u32,
    pub channel_count: u8,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model_id: u32,
    pub input_data: Vec<i8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub model_id: u32,
    pub output_logits: Vec<i8>,
    pub execution_cycles: u32,
    pub execution_time_us: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiveMessage {
    SensorData(SensorFrame),
    RunInference(InferenceRequest),
    InferenceResult(InferenceResponse),
    Ping,
    Pong,
}

impl LiveMessage {
    pub fn encode_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn decode_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
