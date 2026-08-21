//! Live HIL protocol between `embedded-nn-studio` / `enn hil` and a flashed MCU agent.
//!
//! Modeled on `embedded-gui-live`: compact length-prefixed frames with a resync
//! magic and CRC-16 so a constant-memory [`Decoder`] on the MCU can accept
//! partial USB reads without buffering whole JSON documents.
//!
//! Magic bytes are `0xE6 0x4E` (`NN`) so this stream cannot be mistaken for the
//! GUI live protocol (`0xE6 0x71`).
//!
//! ```text
//! +--------+--------+------+-----------+------------------+--------+
//! | 0xE6   | 0x4E   | type | len (u32) | payload (len B)  | crc16  |
//! +--------+--------+------+-----------+------------------+--------+
//!   magic0   magic1   u8     LE          message body       LE
//! ```
//!
//! The CRC-16 (CCITT-FALSE) covers `type`, the 4 length bytes, and the payload.
//! All multi-byte integers are little-endian. Tensor payloads are raw `i8`.
//! Sensor samples are little-endian `f32` values.

#![forbid(unsafe_code)]

/// Protocol version. Bump on any wire-incompatible change; both sides compare
/// this in the [`Msg::Hello`] / [`Msg::Ready`] handshake.
pub const PROTO_VERSION: u16 = 1;

const MAGIC0: u8 = 0xE6;
const MAGIC1: u8 = 0x4E;

/// Bytes preceding every payload: `magic0, magic1, type, len(4)`.
pub const HEADER_LEN: usize = 7;
/// Trailing CRC-16 bytes.
pub const TRAILER_LEN: usize = 2;
/// Total framing overhead around a payload.
pub const FRAME_OVERHEAD: usize = HEADER_LEN + TRAILER_LEN;

const T_HELLO: u8 = 0x01;
const T_RUN_INFERENCE: u8 = 0x02;
const T_PING: u8 = 0x05;

const T_READY: u8 = 0x81;
const T_INFERENCE_RESULT: u8 = 0x82;
const T_SENSOR: u8 = 0x83;
const T_NACK: u8 = 0x84;
const T_PONG: u8 = 0x85;

/// Fixed fields of [`Msg::Hello`] / the first part of [`Msg::Ready`].
pub const HELLO_PAYLOAD_LEN: usize = 14;
/// proto + model_id + input_len + output_len + max_payload + cpu_hz
pub const READY_PAYLOAD_LEN: usize = 22;
/// seq + model_id before the raw i8 input vector.
pub const RUN_INFERENCE_HEADER: usize = 8;
/// seq + model_id + cycles + time_us before the raw i8 logits.
pub const INFERENCE_RESULT_HEADER: usize = 16;
/// timestamp_ms + channel_count before f32 samples.
pub const SENSOR_HEADER: usize = 5;

/// A decoded protocol message. Tensor and sensor payloads borrow from the
/// decoder (or caller) buffer to stay allocation-free on the device.
#[derive(Debug, Clone, PartialEq)]
pub enum Msg<'a> {
    /// Host handshake: protocol version and the model shape it intends to run.
    Hello {
        proto: u16,
        model_id: u32,
        input_len: u32,
        output_len: u32,
    },
    /// Host request to run integer `predict` on `input`.
    RunInference {
        seq: u32,
        model_id: u32,
        input: &'a [u8],
    },
    Ping,
    /// Device handshake reply. `max_payload` bounds a single frame body;
    /// `cpu_hz` is 0 if cycle-to-time conversion is unknown.
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
        logits: &'a [u8],
    },
    SensorFrame {
        timestamp_ms: u32,
        channel_count: u8,
        values: &'a [u8],
    },
    Nack { seq: u32, code: u16 },
    Pong,
}

/// Well-known [`Msg::Nack`] codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum NackCode {
    BadProto = 1,
    Malformed = 2,
    ModelMismatch = 3,
    BadInputLen = 4,
    InferFailed = 5,
    Overflow = 6,
}

/// Errors produced while encoding a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    BufferTooSmall,
    SensorLenMismatch,
}

/// Errors produced while decoding a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    BadCrc,
    Overflow,
    UnknownType(u8),
    BadLength,
}

impl<'a> Msg<'a> {
    fn type_tag(&self) -> u8 {
        match self {
            Msg::Hello { .. } => T_HELLO,
            Msg::RunInference { .. } => T_RUN_INFERENCE,
            Msg::Ping => T_PING,
            Msg::Ready { .. } => T_READY,
            Msg::InferenceResult { .. } => T_INFERENCE_RESULT,
            Msg::SensorFrame { .. } => T_SENSOR,
            Msg::Nack { .. } => T_NACK,
            Msg::Pong => T_PONG,
        }
    }

    /// Length of the payload (message body) this message encodes to.
    pub fn payload_len(&self) -> usize {
        match self {
            Msg::Hello { .. } => HELLO_PAYLOAD_LEN,
            Msg::RunInference { input, .. } => RUN_INFERENCE_HEADER + input.len(),
            Msg::Ping | Msg::Pong => 0,
            Msg::Ready { .. } => READY_PAYLOAD_LEN,
            Msg::InferenceResult { logits, .. } => INFERENCE_RESULT_HEADER + logits.len(),
            Msg::SensorFrame { values, .. } => SENSOR_HEADER + values.len(),
            Msg::Nack { .. } => 6,
        }
    }

    /// Total bytes this message occupies on the wire including framing.
    pub fn encoded_len(&self) -> usize {
        FRAME_OVERHEAD + self.payload_len()
    }

    /// Encodes this message into `out`, returning the number of bytes written.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, EncodeError> {
        if let Msg::SensorFrame {
            channel_count,
            values,
            ..
        } = self
        {
            if !values.len().is_multiple_of(4) {
                return Err(EncodeError::SensorLenMismatch);
            }
            let samples = values.len() / 4;
            if *channel_count != 0 && samples % usize::from(*channel_count) != 0 {
                return Err(EncodeError::SensorLenMismatch);
            }
        }

        let payload_len = self.payload_len();
        let total = FRAME_OVERHEAD + payload_len;
        if out.len() < total {
            return Err(EncodeError::BufferTooSmall);
        }

        out[0] = MAGIC0;
        out[1] = MAGIC1;
        out[2] = self.type_tag();
        out[3..7].copy_from_slice(&(payload_len as u32).to_le_bytes());

        let body = &mut out[HEADER_LEN..HEADER_LEN + payload_len];
        self.encode_body(body);

        let crc = crc16(&out[2..HEADER_LEN + payload_len]);
        out[HEADER_LEN + payload_len..total].copy_from_slice(&crc.to_le_bytes());
        Ok(total)
    }

    fn encode_body(&self, body: &mut [u8]) {
        match *self {
            Msg::Hello {
                proto,
                model_id,
                input_len,
                output_len,
            } => {
                body[0..2].copy_from_slice(&proto.to_le_bytes());
                body[2..6].copy_from_slice(&model_id.to_le_bytes());
                body[6..10].copy_from_slice(&input_len.to_le_bytes());
                body[10..14].copy_from_slice(&output_len.to_le_bytes());
            }
            Msg::RunInference {
                seq,
                model_id,
                input,
            } => {
                body[0..4].copy_from_slice(&seq.to_le_bytes());
                body[4..8].copy_from_slice(&model_id.to_le_bytes());
                body[8..8 + input.len()].copy_from_slice(input);
            }
            Msg::Ping | Msg::Pong => {}
            Msg::Ready {
                proto,
                model_id,
                input_len,
                output_len,
                max_payload,
                cpu_hz,
            } => {
                body[0..2].copy_from_slice(&proto.to_le_bytes());
                body[2..6].copy_from_slice(&model_id.to_le_bytes());
                body[6..10].copy_from_slice(&input_len.to_le_bytes());
                body[10..14].copy_from_slice(&output_len.to_le_bytes());
                body[14..18].copy_from_slice(&max_payload.to_le_bytes());
                body[18..22].copy_from_slice(&cpu_hz.to_le_bytes());
            }
            Msg::InferenceResult {
                seq,
                model_id,
                execution_cycles,
                execution_time_us,
                logits,
            } => {
                body[0..4].copy_from_slice(&seq.to_le_bytes());
                body[4..8].copy_from_slice(&model_id.to_le_bytes());
                body[8..12].copy_from_slice(&execution_cycles.to_le_bytes());
                body[12..16].copy_from_slice(&execution_time_us.to_le_bytes());
                body[16..16 + logits.len()].copy_from_slice(logits);
            }
            Msg::SensorFrame {
                timestamp_ms,
                channel_count,
                values,
            } => {
                body[0..4].copy_from_slice(&timestamp_ms.to_le_bytes());
                body[4] = channel_count;
                body[5..5 + values.len()].copy_from_slice(values);
            }
            Msg::Nack { seq, code } => {
                body[0..4].copy_from_slice(&seq.to_le_bytes());
                body[4..6].copy_from_slice(&code.to_le_bytes());
            }
        }
    }
}

fn parse(msg_type: u8, body: &[u8]) -> Result<Msg<'_>, DecodeError> {
    let u16le = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let u32le = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);

    match msg_type {
        T_HELLO => {
            if body.len() != HELLO_PAYLOAD_LEN {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::Hello {
                proto: u16le(&body[0..2]),
                model_id: u32le(&body[2..6]),
                input_len: u32le(&body[6..10]),
                output_len: u32le(&body[10..14]),
            })
        }
        T_RUN_INFERENCE => {
            if body.len() < RUN_INFERENCE_HEADER {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::RunInference {
                seq: u32le(&body[0..4]),
                model_id: u32le(&body[4..8]),
                input: &body[8..],
            })
        }
        T_PING => {
            if !body.is_empty() {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::Ping)
        }
        T_READY => {
            if body.len() != READY_PAYLOAD_LEN {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::Ready {
                proto: u16le(&body[0..2]),
                model_id: u32le(&body[2..6]),
                input_len: u32le(&body[6..10]),
                output_len: u32le(&body[10..14]),
                max_payload: u32le(&body[14..18]),
                cpu_hz: u32le(&body[18..22]),
            })
        }
        T_INFERENCE_RESULT => {
            if body.len() < INFERENCE_RESULT_HEADER {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::InferenceResult {
                seq: u32le(&body[0..4]),
                model_id: u32le(&body[4..8]),
                execution_cycles: u32le(&body[8..12]),
                execution_time_us: u32le(&body[12..16]),
                logits: &body[16..],
            })
        }
        T_SENSOR => {
            if body.len() < SENSOR_HEADER || !(body.len() - SENSOR_HEADER).is_multiple_of(4) {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::SensorFrame {
                timestamp_ms: u32le(&body[0..4]),
                channel_count: body[4],
                values: &body[5..],
            })
        }
        T_NACK => {
            if body.len() != 6 {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::Nack {
                seq: u32le(&body[0..4]),
                code: u16le(&body[4..6]),
            })
        }
        T_PONG => {
            if !body.is_empty() {
                return Err(DecodeError::BadLength);
            }
            Ok(Msg::Pong)
        }
        other => Err(DecodeError::UnknownType(other)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Magic0,
    Magic1,
    Type,
    Len(u8),
    Payload,
    Crc(u8),
}

/// A constant-memory, resynchronizing frame decoder.
///
/// `CAP` is the maximum payload the decoder can assemble. Size it to
/// `max(RUN_INFERENCE_HEADER + input_len, INFERENCE_RESULT_HEADER + output_len)`
/// on the device.
pub struct Decoder<const CAP: usize> {
    state: State,
    msg_type: u8,
    payload_len: u32,
    buf: [u8; CAP],
    got: usize,
    crc_lo: u8,
    ready: bool,
}

impl<const CAP: usize> Default for Decoder<CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize> Decoder<CAP> {
    /// Creates an empty decoder.
    pub const fn new() -> Self {
        Self {
            state: State::Magic0,
            msg_type: 0,
            payload_len: 0,
            buf: [0u8; CAP],
            got: 0,
            crc_lo: 0,
            ready: false,
        }
    }

    fn resync(&mut self) {
        self.state = State::Magic0;
        self.got = 0;
        self.ready = false;
    }

    /// Feeds a single byte. Returns `Ok(true)` when a full, CRC-valid frame has
    /// been assembled (read it with [`Decoder::message`] before the next push).
    pub fn push(&mut self, byte: u8) -> Result<bool, DecodeError> {
        if self.ready {
            self.resync();
        }

        match self.state {
            State::Magic0 => {
                if byte == MAGIC0 {
                    self.state = State::Magic1;
                }
            }
            State::Magic1 => {
                if byte == MAGIC1 {
                    self.state = State::Type;
                } else if byte == MAGIC0 {
                    self.state = State::Magic1;
                } else {
                    self.state = State::Magic0;
                }
            }
            State::Type => {
                self.msg_type = byte;
                self.payload_len = 0;
                self.state = State::Len(0);
            }
            State::Len(i) => {
                self.payload_len |= (byte as u32) << (8 * i as u32);
                if i == 3 {
                    let len = self.payload_len as usize;
                    if len > CAP {
                        self.resync();
                        return Err(DecodeError::Overflow);
                    }
                    self.got = 0;
                    self.state = if len == 0 {
                        State::Crc(0)
                    } else {
                        State::Payload
                    };
                } else {
                    self.state = State::Len(i + 1);
                }
            }
            State::Payload => {
                self.buf[self.got] = byte;
                self.got += 1;
                if self.got == self.payload_len as usize {
                    self.state = State::Crc(0);
                }
            }
            State::Crc(0) => {
                self.crc_lo = byte;
                self.state = State::Crc(1);
            }
            State::Crc(_) => {
                let got_crc = u16::from_le_bytes([self.crc_lo, byte]);
                if got_crc == self.frame_crc() {
                    self.ready = true;
                    return Ok(true);
                } else {
                    self.resync();
                    return Err(DecodeError::BadCrc);
                }
            }
        }
        Ok(false)
    }

    /// Feeds a slice, invoking `on_msg` for each complete frame.
    pub fn feed<F, E>(&mut self, data: &[u8], mut on_msg: F, mut on_err: E)
    where
        F: FnMut(Msg<'_>),
        E: FnMut(DecodeError),
    {
        for &b in data {
            match self.push(b) {
                Ok(true) => match self.message() {
                    Ok(msg) => on_msg(msg),
                    Err(e) => on_err(e),
                },
                Ok(false) => {}
                Err(e) => on_err(e),
            }
        }
    }

    fn frame_crc(&self) -> u16 {
        let mut crc = CRC_INIT;
        crc = crc16_step(crc, self.msg_type);
        for b in self.payload_len.to_le_bytes() {
            crc = crc16_step(crc, b);
        }
        for &b in &self.buf[..self.payload_len as usize] {
            crc = crc16_step(crc, b);
        }
        crc
    }

    /// Parses the frame assembled by the last successful [`Decoder::push`].
    pub fn message(&self) -> Result<Msg<'_>, DecodeError> {
        parse(self.msg_type, &self.buf[..self.payload_len as usize])
    }
}

const CRC_INIT: u16 = 0xFFFF;

#[inline]
fn crc16_step(mut crc: u16, byte: u8) -> u16 {
    crc ^= (byte as u16) << 8;
    let mut i = 0;
    while i < 8 {
        if crc & 0x8000 != 0 {
            crc = (crc << 1) ^ 0x1021;
        } else {
            crc <<= 1;
        }
        i += 1;
    }
    crc
}

/// Computes the CRC-16/CCITT-FALSE over `data`.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = CRC_INIT;
    for &b in data {
        crc = crc16_step(crc, b);
    }
    crc
}

/// Packs `f32` samples into little-endian bytes for [`Msg::SensorFrame`].
pub fn encode_f32_le(values: &[f32], out: &mut [u8]) -> Result<usize, EncodeError> {
    let need = values.len() * 4;
    if out.len() < need {
        return Err(EncodeError::BufferTooSmall);
    }
    for (i, value) in values.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(need)
}

/// Interprets a [`Msg::SensorFrame`] payload as little-endian `f32` samples.
pub fn decode_f32_le(bytes: &[u8], out: &mut [f32]) -> Result<usize, DecodeError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DecodeError::BadLength);
    }
    let n = bytes.len() / 4;
    if out.len() < n {
        return Err(DecodeError::Overflow);
    }
    for i in 0..n {
        out[i] = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn roundtrip(msg: Msg<'_>) {
        let mut buf = vec![0u8; msg.encoded_len()];
        let n = msg.encode(&mut buf).unwrap();
        assert_eq!(n, msg.encoded_len());

        let mut dec = Decoder::<65536>::new();
        let mut got = None;
        for (i, &b) in buf.iter().enumerate() {
            let ready = dec.push(b).unwrap();
            if i + 1 == buf.len() {
                assert!(ready);
            }
            if ready {
                got = Some(owned(&dec.message().unwrap()));
            }
        }
        assert_eq!(got.unwrap(), owned(&msg));
    }

    #[derive(Debug, PartialEq)]
    enum Owned {
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
        InferenceResult {
            seq: u32,
            model_id: u32,
            execution_cycles: u32,
            execution_time_us: u32,
            logits: Vec<u8>,
        },
        Sensor {
            timestamp_ms: u32,
            channel_count: u8,
            values: Vec<u8>,
        },
        Ready {
            proto: u16,
            model_id: u32,
            input_len: u32,
            output_len: u32,
            max_payload: u32,
            cpu_hz: u32,
        },
        Nack {
            seq: u32,
            code: u16,
        },
        Ping,
        Pong,
    }

    fn owned(msg: &Msg<'_>) -> Owned {
        match *msg {
            Msg::Hello {
                proto,
                model_id,
                input_len,
                output_len,
            } => Owned::Hello {
                proto,
                model_id,
                input_len,
                output_len,
            },
            Msg::RunInference {
                seq,
                model_id,
                input,
            } => Owned::RunInference {
                seq,
                model_id,
                input: input.to_vec(),
            },
            Msg::InferenceResult {
                seq,
                model_id,
                execution_cycles,
                execution_time_us,
                logits,
            } => Owned::InferenceResult {
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
            } => Owned::Sensor {
                timestamp_ms,
                channel_count,
                values: values.to_vec(),
            },
            Msg::Ready {
                proto,
                model_id,
                input_len,
                output_len,
                max_payload,
                cpu_hz,
            } => Owned::Ready {
                proto,
                model_id,
                input_len,
                output_len,
                max_payload,
                cpu_hz,
            },
            Msg::Nack { seq, code } => Owned::Nack { seq, code },
            Msg::Ping => Owned::Ping,
            Msg::Pong => Owned::Pong,
        }
    }

    #[test]
    fn roundtrip_control_and_tensor_messages() {
        roundtrip(Msg::Hello {
            proto: PROTO_VERSION,
            model_id: 1,
            input_len: 4,
            output_len: 2,
        });
        roundtrip(Msg::Ready {
            proto: PROTO_VERSION,
            model_id: 1,
            input_len: 4,
            output_len: 2,
            max_payload: 4096,
            cpu_hz: 100_000_000,
        });
        roundtrip(Msg::Ping);
        roundtrip(Msg::Pong);
        roundtrip(Msg::Nack {
            seq: 9,
            code: NackCode::BadInputLen as u16,
        });
        roundtrip(Msg::RunInference {
            seq: 3,
            model_id: 1,
            input: &[1i8 as u8, 2, 253, 4],
        });
        roundtrip(Msg::InferenceResult {
            seq: 3,
            model_id: 1,
            execution_cycles: 1200,
            execution_time_us: 12,
            logits: &[10, 20],
        });
        let mut packed = [0u8; 8];
        encode_f32_le(&[1.5, -2.0], &mut packed).unwrap();
        roundtrip(Msg::SensorFrame {
            timestamp_ms: 50,
            channel_count: 2,
            values: &packed,
        });
    }

    #[test]
    fn decoder_resyncs_after_garbage() {
        let msg = Msg::Pong;
        let mut buf = vec![0u8; msg.encoded_len()];
        msg.encode(&mut buf).unwrap();
        let mut stream: Vec<u8> = vec![0x00, 0xFF, 0xE6, 0x12];
        stream.extend_from_slice(&buf);

        let mut dec = Decoder::<256>::new();
        let mut got_pong = false;
        dec.feed(
            &stream,
            |m| {
                if matches!(m, Msg::Pong) {
                    got_pong = true;
                }
            },
            |_e| {},
        );
        assert!(got_pong);
    }

    #[test]
    fn decoder_reports_bad_crc() {
        let msg = Msg::Ping;
        let mut buf = vec![0u8; msg.encoded_len()];
        msg.encode(&mut buf).unwrap();
        *buf.last_mut().unwrap() ^= 0xFF;
        let mut dec = Decoder::<256>::new();
        let mut errors = 0;
        dec.feed(&buf, |_m| panic!("should not decode"), |_e| errors += 1);
        assert!(errors >= 1);
    }

    #[test]
    fn decoder_overflow_when_payload_exceeds_capacity() {
        let mut dec = Decoder::<8>::new();
        let mut got_overflow = false;
        let header = [MAGIC0, MAGIC1, T_RUN_INFERENCE, 0x00, 0x01, 0x00, 0x00];
        for &b in &header {
            if let Err(DecodeError::Overflow) = dec.push(b) {
                got_overflow = true;
            }
        }
        assert!(got_overflow);
    }

    #[test]
    fn f32_roundtrip() {
        let src = [0.25f32, -8.0, 16.5];
        let mut bytes = [0u8; 12];
        encode_f32_le(&src, &mut bytes).unwrap();
        let mut out = [0f32; 3];
        assert_eq!(decode_f32_le(&bytes, &mut out).unwrap(), 3);
        assert_eq!(out, src);
    }
}
