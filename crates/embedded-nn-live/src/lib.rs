#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
pub mod dataset;
pub mod protocol;

#[cfg(feature = "std")]
pub use dataset::{DatasetParseError, DatasetRecord, parse_jsonl};
pub use protocol::{
    DecodeError, Decoder, EncodeError, FRAME_OVERHEAD, HEADER_LEN, HELLO_PAYLOAD_LEN,
    INFERENCE_RESULT_HEADER, Msg, NackCode, PROTO_VERSION, READY_PAYLOAD_LEN, RUN_INFERENCE_HEADER,
    SENSOR_HEADER, TRAILER_LEN, crc16, decode_f32_le, encode_f32_le,
};

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
pub mod host;

#[cfg(all(feature = "std", target_arch = "wasm32"))]
#[path = "host_wasm.rs"]
pub mod host;
