//! LiteRT AOT Compiler Plugin for `embedded-nn`.
//!
//! Exposes the standard LiteRT Compiler Plugin C ABI (`libLiteRtCompilerPlugin_embedded_nn.so`)
//! allowing Google LiteRT toolchains (`litert-cli`, `ai-edge-torch`) to partition and compile
//! neural network subgraphs into zero-allocation `#![no_std]` Rust and CMSIS-NN C code
//! with static arena scheduling.

use core::ffi::c_char;
use core::ffi::c_void;

/// LiteRT Status codes (matching `litert_common.h`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteRtStatus {
    Ok = 0,
    ErrorInvalidArgument = 1,
    ErrorUnsupported = 2,
    ErrorRuntimeFailure = 3,
    ErrorNotFound = 4,
}

/// LiteRT API Version struct.
#[repr(C)]
pub struct LiteRtApiVersion {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
}

/// Supported hardware accelerator bitmask.
pub const LITERT_HW_ACCELERATOR_CPU: u64 = 1 << 0;
pub const LITERT_HW_ACCELERATOR_NPU: u64 = 1 << 2;

/// Manufacturer identifier string for this plugin.
pub const SOC_MANUFACTURER: &str = "ARM / embedded-nn";

/// Supported SoC models for code generation targets.
pub const SUPPORTED_SOC_MODELS: &[&str] = &[
    "cortex-m33",
    "cortex-m4",
    "cortex-m7",
    "stm32wba65ri",
    "stm32f4",
    "stm32f7",
    "generic-armv8m",
];

/// Opaque Compiler Plugin instance state.
pub struct PluginState {
    pub target_soc: String,
}

/// Compiled result artifact holding generated code and arena layout.
pub struct CompiledArtifact {
    pub bytecode: Vec<u8>,
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtGetCompilerPluginVersion(
    version: *mut LiteRtApiVersion,
) -> LiteRtStatus {
    if version.is_null() {
        return LiteRtStatus::ErrorInvalidArgument;
    }
    unsafe {
        (*version).major = 1;
        (*version).minor = 0;
        (*version).patch = 0;
    }
    LiteRtStatus::Ok
}

#[unsafe(no_mangle)]
pub extern "C" fn LiteRtGetCompilerPluginSocManufacturer() -> *const c_char {
    // Static nul-terminated C string
    c"ARM / embedded-nn".as_ptr()
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtCreateCompilerPlugin(
    _context: *const c_void,
    plugin: *mut *mut c_void,
    _env_options: *const c_void,
    _options: *const c_void,
) -> LiteRtStatus {
    if plugin.is_null() {
        return LiteRtStatus::ErrorInvalidArgument;
    }
    let state = Box::new(PluginState {
        target_soc: "cortex-m33".into(),
    });
    unsafe {
        *plugin = Box::into_raw(state) as *mut c_void;
    }
    LiteRtStatus::Ok
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtDestroyCompilerPlugin(plugin: *mut c_void) {
    if !plugin.is_null() {
        unsafe {
            let _ = Box::from_raw(plugin as *mut PluginState);
        }
    }
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtGetCompilerPluginSupportedHardware(
    _plugin: *mut c_void,
    supported_hw: *mut u64,
) -> LiteRtStatus {
    if supported_hw.is_null() {
        return LiteRtStatus::ErrorInvalidArgument;
    }
    unsafe {
        *supported_hw = LITERT_HW_ACCELERATOR_CPU | LITERT_HW_ACCELERATOR_NPU;
    }
    LiteRtStatus::Ok
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtGetNumCompilerPluginSupportedSocModels(
    _plugin: *mut c_void,
    num_models: *mut usize,
) -> LiteRtStatus {
    if num_models.is_null() {
        return LiteRtStatus::ErrorInvalidArgument;
    }
    unsafe {
        *num_models = SUPPORTED_SOC_MODELS.len();
    }
    LiteRtStatus::Ok
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtGetCompilerPluginSupportedSocModel(
    _plugin: *mut c_void,
    soc_model_idx: usize,
    soc_model_name: *mut *const c_char,
) -> LiteRtStatus {
    if soc_model_name.is_null() || soc_model_idx >= SUPPORTED_SOC_MODELS.len() {
        return LiteRtStatus::ErrorInvalidArgument;
    }
    let c_str = match soc_model_idx {
        0 => c"cortex-m33",
        1 => c"cortex-m4",
        2 => c"cortex-m7",
        3 => c"stm32wba65ri",
        4 => c"stm32f4",
        5 => c"stm32f7",
        _ => c"generic-armv8m",
    };
    unsafe {
        *soc_model_name = c_str.as_ptr();
    }
    LiteRtStatus::Ok
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtDestroyCompiledResult(result: *mut c_void) {
    if !result.is_null() {
        unsafe {
            let _ = Box::from_raw(result as *mut CompiledArtifact);
        }
    }
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LiteRtGetCompiledResultByteCode(
    result: *mut c_void,
    byte_code_idx: usize,
    byte_code: *mut *const c_void,
    byte_code_size: *mut usize,
) -> LiteRtStatus {
    if result.is_null() || byte_code.is_null() || byte_code_size.is_null() || byte_code_idx != 0 {
        return LiteRtStatus::ErrorInvalidArgument;
    }
    let artifact = unsafe { &*(result as *const CompiledArtifact) };
    unsafe {
        *byte_code = artifact.bytecode.as_ptr() as *const c_void;
        *byte_code_size = artifact.bytecode.len();
    }
    LiteRtStatus::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_litert_plugin_lifecycle_and_metadata() {
        let mut version = LiteRtApiVersion {
            major: 0,
            minor: 0,
            patch: 0,
        };
        assert_eq!(
            unsafe { LiteRtGetCompilerPluginVersion(&mut version) },
            LiteRtStatus::Ok
        );
        assert_eq!(version.major, 1);

        let manufacturer = LiteRtGetCompilerPluginSocManufacturer();
        assert!(!manufacturer.is_null());

        let mut plugin: *mut c_void = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                LiteRtCreateCompilerPlugin(
                    std::ptr::null(),
                    &mut plugin,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            LiteRtStatus::Ok
        );
        assert!(!plugin.is_null());

        let mut hw: u64 = 0;
        assert_eq!(
            unsafe { LiteRtGetCompilerPluginSupportedHardware(plugin, &mut hw) },
            LiteRtStatus::Ok
        );
        assert!(hw & LITERT_HW_ACCELERATOR_CPU != 0);

        let mut num_models = 0usize;
        assert_eq!(
            unsafe { LiteRtGetNumCompilerPluginSupportedSocModels(plugin, &mut num_models) },
            LiteRtStatus::Ok
        );
        assert_eq!(num_models, SUPPORTED_SOC_MODELS.len());

        let mut model_name: *const c_char = std::ptr::null();
        assert_eq!(
            unsafe { LiteRtGetCompilerPluginSupportedSocModel(plugin, 3, &mut model_name) },
            LiteRtStatus::Ok
        );
        assert!(!model_name.is_null());

        unsafe {
            LiteRtDestroyCompilerPlugin(plugin);
        }
    }
}
