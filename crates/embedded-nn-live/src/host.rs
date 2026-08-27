//! Host USB bulk transport for the live HIL protocol.
//!
//! The device enumerates as a vendor-specific WinUSB function (not CDC). Frames
//! use [`crate::Msg`] with 512-byte bulk MPS chunking, matching the proven
//! STM32WBA USB-HS agent layout.

use std::{
    collections::VecDeque,
    io::{Read, Write},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use nusb::{
    MaybeFuture,
    descriptors::TransferType,
    io::{EndpointRead, EndpointWrite},
    transfer::{Bulk, Direction, In, Out},
};

use crate::{DecodeError, Decoder, EncodeError, Msg, PROTO_VERSION};

/// pid.codes vendor ID shared with the GUI live agent.
pub const AGENT_VID: u16 = 0x1209;
/// Distinct from the GUI display agent (`0xE611`).
pub const AGENT_PID: u16 = 0xE612;
/// Native USB-HS bulk max packet size used by the WBA agent.
pub const USB_MPS: usize = 512;
/// Default decoder / payload bound for TinyML tensors.
pub const DEFAULT_MAX_PAYLOAD_SIZE: usize = 16 * 1024;
const HOST_DECODER_CAP: usize = 64 * 1024;
const VENDOR_CLASS: u8 = 0xFF;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("USB device enumeration failed: {0}")]
    Enumeration(nusb::Error),
    #[error("USB operation failed: {0}")]
    Usb(#[from] nusb::Error),
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

/// Stable identity and descriptive metadata for a USB device.
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
        nusb::list_devices()
            .wait()
            .map_err(TransportError::Enumeration)
            .map(|devices| {
                devices
                    .map(|device| Self::from_device_info(&device))
                    .collect()
            })
    }

    pub fn enumerate_nn_agents() -> Result<Vec<Self>, TransportError> {
        Ok(Self::enumerate_devices()?
            .into_iter()
            .filter(|device| device.vendor_id == AGENT_VID && device.product_id == AGENT_PID)
            .collect())
    }

    pub fn list_devices() -> Vec<String> {
        Self::enumerate_nn_agents()
            .unwrap_or_default()
            .into_iter()
            .map(|device| device.display_name())
            .collect()
    }

    pub fn display_name(&self) -> String {
        self.product_string.clone().unwrap_or_else(|| {
            format!(
                "{:04x}:{:04x} ({})",
                self.vendor_id,
                self.product_id,
                self.stable_id()
            )
        })
    }

    pub fn stable_id(&self) -> String {
        if let Some(serial) = &self.serial_number {
            return serial.clone();
        }
        match (&self.bus_id, self.device_address) {
            (Some(bus), Some(address)) => format!("{bus}:{address}"),
            _ => format!("{:04x}:{:04x}", self.vendor_id, self.product_id),
        }
    }

    pub fn open(&self) -> Result<UsbTransport, TransportError> {
        self.open_with_config(UsbTransportConfig::default())
    }

    pub fn open_with_config(
        &self,
        config: UsbTransportConfig,
    ) -> Result<UsbTransport, TransportError> {
        config.validate()?;
        let info = nusb::list_devices()
            .wait()
            .map_err(TransportError::Enumeration)?
            .find(|device| self.matches(device))
            .ok_or(TransportError::DeviceNotFound)?;
        let device = info.open().wait()?;

        let endpoint_set = find_endpoint_set(&device, &config)?;
        let interface = device
            .detach_and_claim_interface(endpoint_set.interface_number)
            .wait()?;
        if endpoint_set.alternate_setting != interface.get_alt_setting() {
            interface
                .set_alt_setting(endpoint_set.alternate_setting)
                .wait()?;
        }

        let reader = interface
            .endpoint::<Bulk, In>(endpoint_set.bulk_in)?
            .reader(config.transfer_size)
            .with_read_timeout(config.timeout);
        let writer = interface
            .endpoint::<Bulk, Out>(endpoint_set.bulk_out)?
            .writer(config.transfer_size)
            .with_write_timeout(config.timeout);

        Ok(UsbTransport {
            reader,
            writer,
            decoder: Box::new(Decoder::new()),
            max_payload_size: config.max_payload_size,
        })
    }

    fn from_device_info(device: &nusb::DeviceInfo) -> Self {
        Self {
            vendor_id: device.vendor_id(),
            product_id: device.product_id(),
            serial_number: device.serial_number().map(str::to_owned),
            product_string: device.product_string().map(str::to_owned),
            manufacturer_string: device.manufacturer_string().map(str::to_owned),
            bus_id: platform_bus_id(device),
            device_address: platform_device_address(device),
            port_chain: platform_port_chain(device),
            legacy_product_selector: None,
        }
    }

    fn matches(&self, device: &nusb::DeviceInfo) -> bool {
        if let Some(name) = &self.legacy_product_selector {
            return device.product_string() == Some(name)
                || Self::from_device_info(device).stable_id() == *name;
        }
        if device.vendor_id() != self.vendor_id || device.product_id() != self.product_id {
            return false;
        }
        if let Some(serial) = &self.serial_number {
            return device.serial_number() == Some(serial);
        }
        if let Some(bus_id) = &self.bus_id {
            if platform_bus_id(device).as_deref() != Some(bus_id) {
                return false;
            }
            if !self.port_chain.is_empty() {
                return platform_port_chain(device) == self.port_chain;
            }
            if let Some(address) = self.device_address {
                return platform_device_address(device) == Some(address);
            }
        }
        true
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn platform_bus_id(device: &nusb::DeviceInfo) -> Option<String> {
    Some(device.bus_id().to_owned())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn platform_bus_id(_: &nusb::DeviceInfo) -> Option<String> {
    None
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn platform_device_address(device: &nusb::DeviceInfo) -> Option<u8> {
    Some(device.device_address())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn platform_device_address(_: &nusb::DeviceInfo) -> Option<u8> {
    None
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn platform_port_chain(device: &nusb::DeviceInfo) -> Vec<u8> {
    device.port_chain().to_vec()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn platform_port_chain(_: &nusb::DeviceInfo) -> Vec<u8> {
    Vec::new()
}

#[derive(Debug, Clone)]
pub struct UsbTransportConfig {
    pub interface_number: Option<u8>,
    pub bulk_in_endpoint: Option<u8>,
    pub bulk_out_endpoint: Option<u8>,
    pub timeout: Duration,
    pub max_payload_size: usize,
    pub transfer_size: usize,
}

impl Default for UsbTransportConfig {
    fn default() -> Self {
        Self {
            interface_number: Some(0),
            bulk_in_endpoint: None,
            bulk_out_endpoint: None,
            timeout: Duration::from_secs(2),
            max_payload_size: DEFAULT_MAX_PAYLOAD_SIZE,
            transfer_size: USB_MPS,
        }
    }
}

impl UsbTransportConfig {
    fn validate(&self) -> Result<(), TransportError> {
        if self.max_payload_size == 0 || self.max_payload_size > HOST_DECODER_CAP {
            return Err(TransportError::InvalidConfig(
                "max_payload_size must be in 1..=64KiB",
            ));
        }
        if self.transfer_size == 0 {
            return Err(TransportError::InvalidConfig(
                "transfer_size must be non-zero",
            ));
        }
        if self.timeout.is_zero() {
            return Err(TransportError::InvalidConfig("timeout must be non-zero"));
        }
        if self
            .bulk_in_endpoint
            .is_some_and(|address| address & 0x80 == 0)
        {
            return Err(TransportError::InvalidConfig(
                "bulk_in_endpoint must be an IN address",
            ));
        }
        if self
            .bulk_out_endpoint
            .is_some_and(|address| address & 0x80 != 0)
        {
            return Err(TransportError::InvalidConfig(
                "bulk_out_endpoint must be an OUT address",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct EndpointSet {
    interface_number: u8,
    alternate_setting: u8,
    bulk_in: u8,
    bulk_out: u8,
    class: u8,
}

fn find_endpoint_set(
    device: &nusb::Device,
    config: &UsbTransportConfig,
) -> Result<EndpointSet, TransportError> {
    let active = device
        .active_configuration()
        .map_err(|_| TransportError::EndpointsNotFound)?;
    let mut candidates = Vec::new();

    for descriptor in active.interface_alt_settings() {
        if config
            .interface_number
            .is_some_and(|number| number != descriptor.interface_number())
        {
            continue;
        }

        let mut bulk_in = None;
        let mut bulk_out = None;
        for endpoint in descriptor
            .endpoints()
            .filter(|endpoint| endpoint.transfer_type() == TransferType::Bulk)
        {
            match endpoint.direction() {
                Direction::In
                    if config
                        .bulk_in_endpoint
                        .is_none_or(|address| endpoint.address() == address) =>
                {
                    bulk_in = Some(endpoint.address())
                }
                Direction::Out
                    if config
                        .bulk_out_endpoint
                        .is_none_or(|address| endpoint.address() == address) =>
                {
                    bulk_out = Some(endpoint.address())
                }
                _ => {}
            }
        }

        if let (Some(bulk_in), Some(bulk_out)) = (bulk_in, bulk_out) {
            candidates.push(EndpointSet {
                interface_number: descriptor.interface_number(),
                alternate_setting: descriptor.alternate_setting(),
                bulk_in,
                bulk_out,
                class: descriptor.class(),
            });
        }
    }

    candidates
        .into_iter()
        .max_by_key(|candidate| {
            let mut score = 0i32;
            if candidate.interface_number == 0 {
                score += 2;
            }
            if candidate.class == VENDOR_CLASS {
                score += 4;
            }
            score
        })
        .ok_or(TransportError::EndpointsNotFound)
}

/// Owned wire message used on the host (tensors copied out of the decoder).
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
            Msg::Nack { seq, code } => Self::Nack { seq, code },
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

    pub fn as_msg(&self) -> Msg<'_> {
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

    pub fn encode_to_vec(&self) -> Result<Vec<u8>, EncodeError> {
        let msg = self.as_msg();
        let mut buf = vec![0u8; msg.encoded_len()];
        let n = msg.encode(&mut buf)?;
        buf.truncate(n);
        Ok(buf)
    }
}

pub trait Transport {
    fn send(&mut self, message: &OwnedMsg) -> Result<(), TransportError>;
    fn receive(&mut self) -> Result<OwnedMsg, TransportError>;
}

pub struct UsbTransport {
    reader: EndpointRead<Bulk>,
    writer: EndpointWrite<Bulk>,
    decoder: Box<Decoder<HOST_DECODER_CAP>>,
    max_payload_size: usize,
}

impl Transport for UsbTransport {
    fn send(&mut self, message: &OwnedMsg) -> Result<(), TransportError> {
        write_owned(&mut self.writer, message, self.max_payload_size)?;
        self.writer.flush()?;
        Ok(())
    }

    fn receive(&mut self) -> Result<OwnedMsg, TransportError> {
        read_owned(&mut self.reader, &mut self.decoder, self.max_payload_size)
    }
}

pub fn write_owned(
    writer: &mut impl Write,
    message: &OwnedMsg,
    max_payload_size: usize,
) -> Result<(), TransportError> {
    let frame = message.encode_to_vec().map_err(TransportError::Encode)?;
    let payload = frame.len().saturating_sub(crate::FRAME_OVERHEAD);
    if payload > max_payload_size {
        return Err(TransportError::Oversized {
            actual: payload,
            max: max_payload_size,
        });
    }
    for chunk in frame.chunks(USB_MPS) {
        writer.write_all(chunk)?;
    }
    Ok(())
}

pub fn read_owned<const CAP: usize>(
    reader: &mut impl Read,
    decoder: &mut Decoder<CAP>,
    max_payload_size: usize,
) -> Result<OwnedMsg, TransportError> {
    let mut packet = [0u8; USB_MPS];
    loop {
        let n = reader.read(&mut packet).map_err(map_read_error)?;
        if n == 0 {
            return Err(TransportError::Disconnected);
        }
        for &byte in &packet[..n] {
            match decoder.push(byte) {
                Ok(true) => {
                    let msg = decoder.message().map_err(TransportError::Decode)?;
                    if msg.payload_len() > max_payload_size {
                        return Err(TransportError::Oversized {
                            actual: msg.payload_len(),
                            max: max_payload_size,
                        });
                    }
                    return Ok(OwnedMsg::from_msg(msg));
                }
                Ok(false) => {}
                Err(error) => return Err(TransportError::Decode(error)),
            }
        }
    }
}

fn map_read_error(error: std::io::Error) -> TransportError {
    if error.kind() == std::io::ErrorKind::TimedOut {
        TransportError::Timeout
    } else if error.kind() == std::io::ErrorKind::UnexpectedEof {
        TransportError::Disconnected
    } else {
        TransportError::Io(error)
    }
}

/// Sends `Hello` and waits for a matching `Ready`.
pub fn handshake(
    transport: &mut impl Transport,
    model_id: u32,
    input_len: u32,
    output_len: u32,
) -> Result<OwnedMsg, TransportError> {
    transport.send(&OwnedMsg::Hello {
        proto: PROTO_VERSION,
        model_id,
        input_len,
        output_len,
    })?;
    loop {
        match transport.receive()? {
            ready @ OwnedMsg::Ready {
                proto,
                model_id: ready_model,
                input_len: ready_in,
                output_len: ready_out,
                ..
            } => {
                if proto != PROTO_VERSION {
                    return Err(TransportError::Handshake("protocol version mismatch"));
                }
                if model_id != 0 && ready_model != model_id {
                    return Err(TransportError::Handshake("model_id mismatch"));
                }
                if input_len != 0 && ready_in != input_len {
                    return Err(TransportError::Handshake("tensor length mismatch"));
                }
                if output_len != 0 && ready_out != output_len {
                    return Err(TransportError::Handshake("tensor length mismatch"));
                }
                return Ok(ready);
            }
            OwnedMsg::SensorFrame { .. } => continue,
            OwnedMsg::Nack { seq, code } => return Err(TransportError::Nack { seq, code }),
            _ => return Err(TransportError::Handshake("expected Ready")),
        }
    }
}

struct MemoryQueue {
    messages: Mutex<VecDeque<OwnedMsg>>,
    available: Condvar,
}

pub struct InMemoryTransport {
    incoming: Arc<MemoryQueue>,
    outgoing: Arc<MemoryQueue>,
    timeout: Duration,
}

pub fn in_memory_transport_pair(timeout: Duration) -> (InMemoryTransport, InMemoryTransport) {
    let first = Arc::new(MemoryQueue {
        messages: Mutex::new(VecDeque::new()),
        available: Condvar::new(),
    });
    let second = Arc::new(MemoryQueue {
        messages: Mutex::new(VecDeque::new()),
        available: Condvar::new(),
    });
    (
        InMemoryTransport {
            incoming: first.clone(),
            outgoing: second.clone(),
            timeout,
        },
        InMemoryTransport {
            incoming: second,
            outgoing: first,
            timeout,
        },
    )
}

impl Transport for InMemoryTransport {
    fn send(&mut self, message: &OwnedMsg) -> Result<(), TransportError> {
        self.outgoing
            .messages
            .lock()
            .map_err(|_| TransportError::Disconnected)?
            .push_back(message.clone());
        self.outgoing.available.notify_one();
        Ok(())
    }

    fn receive(&mut self) -> Result<OwnedMsg, TransportError> {
        let queue = self
            .incoming
            .messages
            .lock()
            .map_err(|_| TransportError::Disconnected)?;
        let (mut queue, wait) = self
            .incoming
            .available
            .wait_timeout_while(queue, self.timeout, |messages| messages.is_empty())
            .map_err(|_| TransportError::Disconnected)?;
        queue.pop_front().ok_or_else(|| {
            if wait.timed_out() {
                TransportError::Timeout
            } else {
                TransportError::Disconnected
            }
        })
    }
}

enum WorkerCmd {
    Ping,
    Infer {
        seq: u32,
        model_id: u32,
        input: Vec<u8>,
    },
}

#[derive(Default)]
struct LinkState {
    handshaked: bool,
    alive: bool,
    error: Option<String>,
    ready: Option<OwnedMsg>,
    latest_sensor: Option<OwnedMsg>,
    last_result: Option<OwnedMsg>,
}

struct Shared {
    cmds: Mutex<VecDeque<WorkerCmd>>,
    state: Mutex<LinkState>,
    quit: AtomicBool,
}

/// Background USB worker so egui never blocks on bulk I/O.
pub struct DeviceLink {
    shared: Arc<Shared>,
    device_id: String,
}

impl DeviceLink {
    pub fn connect(device_id: &str) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            cmds: Mutex::new(VecDeque::new()),
            state: Mutex::new(LinkState {
                alive: true,
                ..Default::default()
            }),
            quit: AtomicBool::new(false),
        });
        let worker_shared = Arc::clone(&shared);
        let name = device_id.to_string();
        thread::Builder::new()
            .name("embedded-nn-device-link".into())
            .spawn(move || device_worker(worker_shared, name))
            .map_err(|e| format!("Failed to spawn link thread: {e}"))?;
        Ok(Self {
            shared,
            device_id: device_id.to_string(),
        })
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub fn is_alive(&self) -> bool {
        self.shared.state.lock().map(|s| s.alive).unwrap_or(false)
    }

    pub fn is_handshaked(&self) -> bool {
        self.shared
            .state
            .lock()
            .map(|s| s.handshaked)
            .unwrap_or(false)
    }

    pub fn ready_info(&self) -> Option<OwnedMsg> {
        self.shared.state.lock().ok().and_then(|s| s.ready.clone())
    }

    pub fn take_error(&self) -> Option<String> {
        self.shared
            .state
            .lock()
            .ok()
            .and_then(|mut s| s.error.take())
    }

    pub fn take_sensor(&self) -> Option<OwnedMsg> {
        self.shared
            .state
            .lock()
            .ok()
            .and_then(|mut s| s.latest_sensor.take())
    }

    pub fn take_result(&self) -> Option<OwnedMsg> {
        self.shared
            .state
            .lock()
            .ok()
            .and_then(|mut s| s.last_result.take())
    }

    pub fn ping(&self) {
        if let Ok(mut cmds) = self.shared.cmds.lock() {
            cmds.push_back(WorkerCmd::Ping);
        }
    }

    pub fn infer(&self, seq: u32, model_id: u32, input: Vec<u8>) {
        if let Ok(mut cmds) = self.shared.cmds.lock() {
            cmds.push_back(WorkerCmd::Infer {
                seq,
                model_id,
                input,
            });
        }
    }
}

impl Drop for DeviceLink {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Release);
    }
}

fn fail(shared: &Shared, msg: String) {
    if let Ok(mut state) = shared.state.lock() {
        state.error = Some(msg);
        state.alive = false;
    }
}

fn apply_inbound(shared: &Shared, msg: OwnedMsg) {
    if let Ok(mut state) = shared.state.lock() {
        match msg {
            ready @ OwnedMsg::Ready { .. } => {
                state.handshaked = true;
                state.ready = Some(ready);
            }
            sensor @ OwnedMsg::SensorFrame { .. } => {
                state.latest_sensor = Some(sensor);
            }
            result @ OwnedMsg::InferenceResult { .. } => {
                state.last_result = Some(result);
            }
            pong @ OwnedMsg::Pong => {
                state.last_result = Some(pong);
            }
            OwnedMsg::Nack { seq, code } => {
                state.error = Some(format!("device nack seq={seq} code={code}"));
            }
            _ => {}
        }
    }
}

fn device_worker(shared: Arc<Shared>, device_id: String) {
    let config = UsbTransportConfig {
        timeout: Duration::from_millis(50),
        ..UsbTransportConfig::default()
    };
    let bridge = UsbBridge {
        vendor_id: AGENT_VID,
        product_id: AGENT_PID,
        serial_number: None,
        product_string: None,
        manufacturer_string: None,
        bus_id: None,
        device_address: None,
        port_chain: Vec::new(),
        legacy_product_selector: Some(device_id),
    };
    let mut transport = match bridge.open_with_config(config) {
        Ok(t) => t,
        Err(e) => return fail(&shared, e.to_string()),
    };

    if let Err(e) = transport.send(&OwnedMsg::Hello {
        proto: PROTO_VERSION,
        model_id: 0,
        input_len: 0,
        output_len: 0,
    }) {
        return fail(&shared, format!("write Hello: {e}"));
    }

    while !shared.quit.load(Ordering::Acquire) {
        match transport.receive() {
            Ok(msg) => apply_inbound(&shared, msg),
            Err(TransportError::Timeout) => {}
            Err(e) => return fail(&shared, e.to_string()),
        }

        let cmd = shared.cmds.lock().ok().and_then(|mut q| q.pop_front());
        let Some(cmd) = cmd else {
            continue;
        };
        let send = match cmd {
            WorkerCmd::Ping => transport.send(&OwnedMsg::Ping),
            WorkerCmd::Infer {
                seq,
                model_id,
                input,
            } => transport.send(&OwnedMsg::RunInference {
                seq,
                model_id,
                input,
            }),
        };
        if let Err(e) = send {
            return fail(&shared, e.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn framed_bytes_round_trip_through_chunked_writer() {
        let msg = OwnedMsg::RunInference {
            seq: 3,
            model_id: 1,
            input: vec![1, 2, 253, 4],
        };
        let mut cursor = Cursor::new(Vec::new());
        write_owned(&mut cursor, &msg, 64).unwrap();
        cursor.set_position(0);
        let mut decoder = Decoder::<256>::new();
        let got = read_owned(&mut cursor, &mut decoder, 64).unwrap();
        assert_eq!(got, msg);
    }

    #[test]
    fn memory_transport_hello_ready_and_inference() {
        let (mut host, mut device) = in_memory_transport_pair(Duration::from_millis(50));
        let device_thread = thread::spawn(move || {
            let hello = device.receive().unwrap();
            match hello {
                OwnedMsg::Hello {
                    proto,
                    model_id,
                    input_len,
                    output_len,
                } => device
                    .send(&OwnedMsg::Ready {
                        proto,
                        model_id,
                        input_len,
                        output_len,
                        max_payload: 128,
                        cpu_hz: 1,
                    })
                    .unwrap(),
                other => panic!("{other:?}"),
            }
            match device.receive().unwrap() {
                OwnedMsg::RunInference { seq, input, .. } => device
                    .send(&OwnedMsg::InferenceResult {
                        seq,
                        model_id: 1,
                        execution_cycles: 42,
                        execution_time_us: 1,
                        logits: input,
                    })
                    .unwrap(),
                other => panic!("{other:?}"),
            }
        });

        let ready = handshake(&mut host, 1, 2, 2).unwrap();
        assert!(matches!(ready, OwnedMsg::Ready { model_id: 1, .. }));
        host.send(&OwnedMsg::RunInference {
            seq: 9,
            model_id: 1,
            input: vec![7, 8],
        })
        .unwrap();
        match host.receive().unwrap() {
            OwnedMsg::InferenceResult {
                seq,
                execution_cycles,
                logits,
                ..
            } => {
                assert_eq!(seq, 9);
                assert_eq!(execution_cycles, 42);
                assert_eq!(logits, vec![7, 8]);
            }
            other => panic!("{other:?}"),
        }
        device_thread.join().unwrap();
    }
}
