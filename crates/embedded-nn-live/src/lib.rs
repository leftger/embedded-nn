pub mod protocol;

pub use protocol::{InferenceRequest, InferenceResponse, LiveMessage, SensorFrame};

#[cfg(feature = "std")]
pub mod host {
    use nusb::MaybeFuture;

    pub struct UsbBridge {
        pub device_name: String,
    }

    impl UsbBridge {
        pub fn new(device_name: impl Into<String>) -> Self {
            Self {
                device_name: device_name.into(),
            }
        }

        pub fn list_devices() -> Vec<String> {
            match nusb::list_devices().wait() {
                Ok(devs) => devs
                    .filter_map(|d| d.product_string().map(|s| s.to_string()))
                    .collect(),
                Err(_) => Vec::new(),
            }
        }
    }
}
