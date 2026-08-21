//! USB-independent HIL helpers for the WBA agent.

use embedded_nn_live::{Msg, NackCode, PROTO_VERSION};

use crate::model::SineFc;

pub const MODEL_ID: u32 = 0;
pub const CPU_HZ: u32 = 96_000_000;
pub const DEC_CAP: usize = 2048;

pub fn ready_msg() -> Msg<'static> {
    Msg::Ready {
        proto: PROTO_VERSION,
        model_id: MODEL_ID,
        input_len: SineFc::INPUT_DIM as u32,
        output_len: SineFc::OUTPUT_DIM as u32,
        max_payload: DEC_CAP as u32,
        cpu_hz: CPU_HZ,
    }
}

pub fn hello_acceptable(proto: u16, model_id: u32, input_len: u32, output_len: u32) -> Option<u16> {
    if proto != PROTO_VERSION {
        return Some(NackCode::BadProto as u16);
    }
    if model_id != 0 && model_id != MODEL_ID {
        return Some(NackCode::ModelMismatch as u16);
    }
    if input_len != 0 && input_len != SineFc::INPUT_DIM as u32 {
        return Some(NackCode::BadInputLen as u16);
    }
    if output_len != 0 && output_len != SineFc::OUTPUT_DIM as u32 {
        return Some(NackCode::BadInputLen as u16);
    }
    None
}
