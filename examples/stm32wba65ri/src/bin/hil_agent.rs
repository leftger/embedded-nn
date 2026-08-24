//! USB-HS bulk HIL agent for STM32WBA65RI.
//!
//! Clock tree, HS PHY pins (PD6/PD7), and WinUSB descriptors follow the proven
//! `stm32wba-tftdisplay` studio_agent. Frames use `embedded-nn-live` (magic
//! `0xE6 0x4E`), not the GUI live protocol.

#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;

use embassy_executor::Spawner;
use embassy_stm32::dma;
use embassy_stm32::i2c::{self, Config as I2cConfig, I2c};
use embassy_stm32::rcc::*;
use embassy_stm32::usb::{self, Driver};
use embassy_stm32::{bind_interrupts, peripherals, Config};
use embassy_time::{Duration, Instant, with_timeout};
use embassy_usb::driver::{Endpoint, EndpointIn, EndpointOut};
use embassy_usb::msos::{
    CompatibleIdFeatureDescriptor, PropertyData, RegistryPropertyFeatureDescriptor,
};
use embassy_usb::{Builder, UsbDevice};
use embedded_nn_live::{Decoder, Msg, NackCode};
use lis2de12::{Lis2de12Async, SlaveAddr};
use static_cell::StaticCell;

#[path = "../model.rs"]
mod model;
#[path = "../hil_usb.rs"]
mod hil_usb;
#[path = "../on_device_dsp.rs"]
mod on_device_dsp;

use hil_usb::{DEC_CAP, MODEL_ID, hello_acceptable, ready_msg};

/// WHO_AM_I register address and expected value, used only to probe which of
/// the two possible 7-bit addresses the LIS2DE12 responds on before handing
/// the bus to `Lis2de12Async::new_i2c`. `SlaveAddr::addr()` is private to the
/// `lis2de12` crate, so the raw addresses are duplicated here (datasheet
/// Table 15, 7-bit form).
const WHO_AM_I_REG: u8 = 0x0F;
const WHO_AM_I_VAL: u8 = 0x33;
const LIS2DE12_ADDR_DEFAULT: u8 = 0x18;
const LIS2DE12_ADDR_ALT: u8 = 0x19;

/// How often to sample and push a `Msg::SensorFrame` while idle (no host
/// request pending). Bounds the USB-HS bulk read wait, not the sensor's own
/// ODR, so it also caps how stale a burst of queued host requests can get.
const SENSOR_PERIOD: Duration = Duration::from_hz(50);

bind_interrupts!(struct Irqs {
    USB_OTG_HS => usb::InterruptHandler<peripherals::USB_OTG_HS>;
    I2C1_EV => i2c::EventInterruptHandler<peripherals::I2C1>;
    I2C1_ER => i2c::ErrorInterruptHandler<peripherals::I2C1>;
    GPDMA1_CHANNEL0 => dma::InterruptHandler<peripherals::GPDMA1_CH0>;
    GPDMA1_CHANNEL1 => dma::InterruptHandler<peripherals::GPDMA1_CH1>;
});

const USB_MPS: u16 = 512;
const AGENT_VID: u16 = 0x1209;
const AGENT_PID: u16 = 0xE612;

type UsbDriver = Driver<'static, peripherals::USB_OTG_HS>;

#[embassy_executor::task]
async fn usb_task(mut device: UsbDevice<'static, UsbDriver>) {
    device.run().await;
}

fn cycles_to_us(cycles: u32) -> u32 {
    (cycles as u64 * 1_000_000 / u64::from(hil_usb::CPU_HZ)) as u32
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let mut dsp_features = [0i8; 16];
    let _ = on_device_dsp::first_frame_s8(&[0.1; 48], &mut dsp_features);
    defmt::info!("on-device DSP first bin {}", dsp_features[0]);

    let mut config = Config::default();
    config.rcc.pll1 = Some(Pll {
        source: PllSource::Hsi,
        prediv: PllPreDiv::Div1,
        mul: PllMul::Mul30,
        divr: Some(PllDiv::Div5),
        divq: Some(PllDiv::Div10),
        divp: Some(PllDiv::Div30),
        frac: Some(0),
    });
    config.rcc.sys = Sysclk::Pll1R;
    config.rcc.ahb_pre = AHBPrescaler::Div1;
    config.rcc.apb1_pre = APBPrescaler::Div1;
    config.rcc.apb2_pre = APBPrescaler::Div1;
    config.rcc.voltage_scale = VoltageScale::Range1;
    config.rcc.mux.otghssel = mux::Otghssel::Pll1P;
    let p = embassy_stm32::init(config);

    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    core.DWT.enable_cycle_counter();

    defmt::info!("embedded-nn hil_agent: booting USB-HS bulk");

    // LIS2DE12 accelerometer on I2C1 (PB2 SCL / PB1 SDA), same wiring as the
    // `data_collector` firmware. Async I2C keeps the WHO_AM_I probe and the
    // periodic reads below from blocking the executor that also drives
    // `usb_task`. Streamed as `Msg::SensorFrame` whenever the USB-HS bulk
    // endpoint is otherwise idle.
    //
    // WBA65's I2C peripheral is the "v2" variant, whose async transfers are
    // always DMA-backed internally (`new_no_dma` leaves the DMA channels as
    // `None`, which panics the first time an async read/write is issued).
    let mut i2c_cfg = I2cConfig::default();
    i2c_cfg.sda_pullup = true;
    i2c_cfg.scl_pullup = true;
    let mut i2c = I2c::new(
        p.I2C1, p.PB2, p.PB1, p.GPDMA1_CH0, p.GPDMA1_CH1, Irqs, i2c_cfg,
    );

    let mut who = [0u8; 1];
    let detected_addr = if i2c
        .write_read(LIS2DE12_ADDR_DEFAULT, &[WHO_AM_I_REG], &mut who)
        .await
        .is_ok()
        && who[0] == WHO_AM_I_VAL
    {
        Some(SlaveAddr::Default)
    } else if i2c
        .write_read(LIS2DE12_ADDR_ALT, &[WHO_AM_I_REG], &mut who)
        .await
        .is_ok()
        && who[0] == WHO_AM_I_VAL
    {
        Some(SlaveAddr::Alternative)
    } else {
        None
    };

    let mut accel = match detected_addr {
        Some(addr) => match Lis2de12Async::new_i2c(i2c, addr).await {
            Ok(dev) => {
                defmt::info!("hil_agent: LIS2DE12 detected, sensor streaming enabled");
                Some(dev)
            }
            Err(_) => {
                defmt::warn!("hil_agent: LIS2DE12 init failed, sensor streaming disabled");
                None
            }
        },
        None => {
            defmt::warn!("hil_agent: LIS2DE12 not detected, sensor streaming disabled");
            None
        }
    };

    static EP_OUT: StaticCell<[u8; 1024]> = StaticCell::new();
    let mut driver_config = embassy_stm32::usb::Config::default();
    driver_config.vbus_detection = false;
    let driver = Driver::new_hs(
        p.USB_OTG_HS,
        Irqs,
        p.PD6,
        p.PD7,
        EP_OUT.init([0u8; 1024]),
        driver_config,
    );

    let mut usb_cfg = embassy_usb::Config::new(AGENT_VID, AGENT_PID);
    usb_cfg.manufacturer = Some("embedded-nn");
    usb_cfg.product = Some("embedded-nn-agent");
    usb_cfg.serial_number = Some("wba65-nn");
    usb_cfg.max_power = 500;
    usb_cfg.composite_with_iads = false;
    usb_cfg.device_class = 0x00;
    usb_cfg.device_sub_class = 0x00;
    usb_cfg.device_protocol = 0x00;

    static CFG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static MSOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static CTRL_BUF: StaticCell<[u8; 128]> = StaticCell::new();
    let mut builder = Builder::new(
        driver,
        usb_cfg,
        CFG_DESC.init([0u8; 256]),
        BOS_DESC.init([0u8; 256]),
        MSOS_DESC.init([0u8; 256]),
        CTRL_BUF.init([0u8; 128]),
    );

    builder.msos_descriptor(0x0600_0000, 0x20);
    builder.msos_feature(CompatibleIdFeatureDescriptor::new("WINUSB", ""));
    builder.msos_feature(RegistryPropertyFeatureDescriptor::new(
        "DeviceInterfaceGUIDs",
        PropertyData::RegMultiSz(&["{B3E91C44-A7D1-4F8E-9C22-77E1D0A91B20}"]),
    ));

    let mut function = builder.function(0xFF, 0xFF, 0xFF);
    let mut interface = function.interface();
    let mut alt = interface.alt_setting(0xFF, 0xFF, 0xFF, None);
    let mut rx = alt.endpoint_bulk_out(None, USB_MPS);
    let mut tx = alt.endpoint_bulk_in(None, USB_MPS);
    drop(alt);
    drop(interface);
    drop(function);

    spawner.spawn(usb_task(builder.build()).expect("spawn usb_task"));

    static DEC: StaticCell<Decoder<DEC_CAP>> = StaticCell::new();
    let dec = DEC.init(Decoder::new());
    let mut packet = [0u8; USB_MPS as usize];
    let mut encode_buf = [0u8; DEC_CAP + embedded_nn_live::FRAME_OVERHEAD];
    let mut logits = [0u8; 64];
    let mut input = [0i8; 64];

    loop {
        rx.wait_enabled().await;
        defmt::info!("hil_agent: host connected");

        loop {
            let n = match with_timeout(SENSOR_PERIOD, rx.read(&mut packet)).await {
                Ok(Ok(n)) => n,
                Ok(Err(_)) => {
                    defmt::warn!("hil_agent: endpoint error, awaiting reconnect");
                    break;
                }
                Err(_timeout) => {
                    if let Some(dev) = accel.as_mut() {
                        if let Ok(g) = dev.read_g().await {
                            let mut values = [0u8; 12];
                            if embedded_nn_live::encode_f32_le(&[g.x, g.y, g.z], &mut values)
                                .is_ok()
                            {
                                let frame = Msg::SensorFrame {
                                    timestamp_ms: Instant::now().as_millis() as u32,
                                    channel_count: 3,
                                    values: &values,
                                };
                                if let Ok(len) = frame.encode(&mut encode_buf) {
                                    let _ = tx.write_transfer(&encode_buf[..len], true).await;
                                }
                            }
                        }
                    }
                    continue;
                }
            };
            for &b in &packet[..n] {
                match dec.push(b) {
                    Ok(true) => {
                        let reply = match dec.message() {
                            Ok(Msg::Hello {
                                proto,
                                model_id,
                                input_len,
                                output_len,
                            }) => {
                                if let Some(code) =
                                    hello_acceptable(proto, model_id, input_len, output_len)
                                {
                                    Some(Msg::Nack { seq: 0, code })
                                } else {
                                    Some(ready_msg())
                                }
                            }
                            Ok(Msg::Ping) => Some(Msg::Pong),
                            Ok(Msg::RunInference {
                                seq,
                                model_id,
                                input: raw,
                            }) => {
                                if model_id != MODEL_ID {
                                    Some(Msg::Nack {
                                        seq,
                                        code: NackCode::ModelMismatch as u16,
                                    })
                                } else if raw.len() != model::SineFc::INPUT_DIM {
                                    Some(Msg::Nack {
                                        seq,
                                        code: NackCode::BadInputLen as u16,
                                    })
                                } else if raw.len() > input.len() || model::SineFc::OUTPUT_DIM > logits.len()
                                {
                                    Some(Msg::Nack {
                                        seq,
                                        code: NackCode::Overflow as u16,
                                    })
                                } else {
                                    for (dst, src) in input.iter_mut().zip(raw.iter()) {
                                        *dst = *src as i8;
                                    }
                                    let mut arena = [0u8; model::SineFc::ARENA_SIZE];
                                    let started = cortex_m::peripheral::DWT::cycle_count();
                                    match model::SineFc::predict(
                                        &input[..model::SineFc::INPUT_DIM],
                                        &mut arena,
                                    ) {
                                        Ok(output) => {
                                            let cycles = cortex_m::peripheral::DWT::cycle_count()
                                                .wrapping_sub(started);
                                            for (dst, src) in logits.iter_mut().zip(output.iter()) {
                                                *dst = *src as u8;
                                            }
                                            Some(Msg::InferenceResult {
                                                seq,
                                                model_id,
                                                execution_cycles: cycles,
                                                execution_time_us: cycles_to_us(cycles),
                                                logits: &logits[..output.len()],
                                            })
                                        }
                                        Err(_) => Some(Msg::Nack {
                                            seq,
                                            code: NackCode::InferFailed as u16,
                                        }),
                                    }
                                }
                            }
                            Ok(_) => None,
                            Err(_) => Some(Msg::Nack {
                                seq: 0,
                                code: NackCode::Malformed as u16,
                            }),
                        };

                        if let Some(msg) = reply {
                            if let Ok(len) = msg.encode(&mut encode_buf) {
                                let _ = tx.write_transfer(&encode_buf[..len], true).await;
                            }
                        }
                    }
                    Ok(false) => {}
                    Err(_) => {}
                }
            }
        }
    }
}
