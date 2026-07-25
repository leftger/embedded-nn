//! Activation functions for quantized tensors.

use crate::support::{clamp, requantize};
use crate::types::Activation;

/// Sigmoid and Tanh 256-element lookup table (Q0.16 format).
pub const SIGMOID_TABLE_UINT16: [u16; 256] = [
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149, 40794, 41432, 42064, 42688,
    43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409,
    51865, 52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433,
    57724, 58007, 58280, 58544, 58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110,
    61279, 61441, 61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886, 62990, 63090, 63186,
    63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835, 63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308,
    64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900,
    64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208,
    65221, 65234, 65246, 65258, 65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360, 65367,
    65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425, 65429, 65433, 65438, 65442, 65445, 65449,
    65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
    65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513,
    65514, 65515, 65516, 65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524,
    65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529, 65529, 65529, 65530, 65530,
    65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533,
    65533, 65533, 65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65535,
];

/// In-place ReLU for int8 buffer (`data[i] = max(data[i], 0)`).
pub fn relu_s8(data: &mut [i8]) {
    for val in data.iter_mut() {
        if *val < 0 {
            *val = 0;
        }
    }
}

/// In-place ReLU6 for int8 buffer (`data[i] = min(max(data[i], 0), 6)`).
pub fn relu6_s8(data: &mut [i8]) {
    for val in data.iter_mut() {
        let mut ip = *val as i32;
        if ip < 0 {
            ip = 0;
        }
        if ip > 6 {
            ip = 6;
        }
        *val = ip as i8;
    }
}

/// In-place generic activation clipping for int8 buffer (`data[i] = clamp(data[i], act.min, act.max)`).
pub fn activation_s8(data: &mut [i8], act: Activation) {
    for val in data.iter_mut() {
        let clamped = clamp(*val as i32, act.min, act.max);
        *val = clamped as i8;
    }
}

/// In-place ReLU for int16 buffer.
pub fn relu_s16(data: &mut [i16]) {
    for val in data.iter_mut() {
        if *val < 0 {
            *val = 0;
        }
    }
}

/// In-place generic activation clipping for int16 buffer.
pub fn activation_s16(data: &mut [i16], act: Activation) {
    for val in data.iter_mut() {
        let clamped = clamp(*val as i32, act.min, act.max);
        *val = clamped as i16;
    }
}

/// LeakyReLU activation for int8 buffer.
pub fn leaky_relu_s8(
    input: &[i8],
    output: &mut [i8],
    alpha_mult: i32,
    alpha_shift: i32,
    input_offset: i32,
    output_offset: i32,
) {
    let size = input.len().min(output.len());
    for i in 0..size {
        let val = input[i] as i32 + input_offset;
        let res = if val < 0 {
            requantize(val, alpha_mult, alpha_shift) + output_offset
        } else {
            val + output_offset
        };
        output[i] = clamp(res, i8::MIN as i32, i8::MAX as i32) as i8;
    }
}

/// Sigmoid activation for int8 tensors using direct lookup table.
pub fn sigmoid_s8(input: &[i8], output: &mut [i8]) {
    let size = input.len().min(output.len());
    for i in 0..size {
        let val = input[i] as i32;
        let abs_val = val.abs();
        let idx = clamp(abs_val, 0, 255) as usize;
        let lut_val = SIGMOID_TABLE_UINT16[idx] as u32;

        let q0_16 = if val >= 0 {
            lut_val
        } else {
            65535 - lut_val
        };

        let s8_val = ((q0_16 as i32) >> 8) - 128;
        output[i] = clamp(s8_val, -128, 127) as i8;
    }
}

/// Tanh activation for int8 tensors using lookup table.
pub fn tanh_s8(input: &[i8], output: &mut [i8]) {
    let size = input.len().min(output.len());
    for i in 0..size {
        let val = input[i] as i32;
        let abs_val = val.abs();
        let idx = clamp(abs_val * 2, 0, 255) as usize;
        let lut_val = SIGMOID_TABLE_UINT16[idx] as i32;
        let res = ((lut_val - 32768) * 2) >> 8;
        let res_signed = if val >= 0 { res } else { -res };
        output[i] = clamp(res_signed, -128, 127) as i8;
    }
}

/// Sigmoid activation for int16 tensors.
pub fn sigmoid_s16(input: &[i16], output: &mut [i16], left_shift: i32) {
    let size = input.len().min(output.len());
    let abs_input_shift = 9u32;
    let max_saturation = (0x7FFF << 10) as u32;
    let input_multiplier = if left_shift < 0 { 3 } else { 3 << left_shift };
    let abs_left_shift = if left_shift < 0 { -left_shift as u32 } else { 0 };
    let rounding = if abs_left_shift > 0 { 1 << (abs_left_shift - 1) } else { 0 };

    for i in 0..size {
        let input_data = ((input[i] as i32) * input_multiplier + rounding) >> abs_left_shift;
        let abs_input_data = input_data.unsigned_abs();
        let uh = (abs_input_data >> abs_input_shift) as usize;

        let result = if uh >= 255 {
            max_saturation
        } else {
            let ua = SIGMOID_TABLE_UINT16[uh] as u32;
            let ub = SIGMOID_TABLE_UINT16[uh + 1] as u32;
            let ut = abs_input_data & 0x1ff;
            (ua << abs_input_shift) + ut * (ub - ua)
        };

        let final_val = if input_data >= 0 {
            (result + (1 << 9)) >> 10
        } else {
            ((1u32 << 25) - result + (1 << 9) - 1) >> 10
        };

        output[i] = clamp(final_val as i32, i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Tanh activation for int16 tensors.
pub fn tanh_s16(input: &[i16], output: &mut [i16], left_shift: i32) {
    let size = input.len().min(output.len());
    let abs_input_shift = 8u32;
    let max_saturation = (0xFFFF << 8) as u32;
    let input_multiplier = if left_shift < 0 { 3 } else { 3 << left_shift };
    let abs_left_shift = if left_shift < 0 { -left_shift as u32 } else { 0 };
    let rounding = if abs_left_shift > 0 { 1 << (abs_left_shift - 1) } else { 0 };

    for i in 0..size {
        let input_data = ((input[i] as i32) * input_multiplier + rounding) >> abs_left_shift;
        let abs_input_data = input_data.unsigned_abs();
        let uh = (abs_input_data >> abs_input_shift) as usize;

        let result = if uh >= 255 {
            max_saturation
        } else {
            let ua = SIGMOID_TABLE_UINT16[uh] as u32;
            let ub = SIGMOID_TABLE_UINT16[uh + 1] as u32;
            let ut = abs_input_data & 0x0ff;
            (ua << abs_input_shift) + ut * (ub - ua)
        };

        let pos_val = (((result as i32) - (1 << 23)) + (1 << 7)) >> 8;
        let final_val = if input_data >= 0 { pos_val } else { -pos_val };

        output[i] = clamp(final_val, i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_s8() {
        let mut data = [-5i8, 0, 10, -128, 127];
        relu_s8(&mut data);
        assert_eq!(data, [0, 0, 10, 0, 127]);
    }

    #[test]
    fn test_relu6_s8() {
        let mut data = [-5i8, 0, 4, 6, 10];
        relu6_s8(&mut data);
        assert_eq!(data, [0, 0, 4, 6, 6]);
    }

    #[test]
    fn test_sigmoid_s8() {
        let input = [0i8, 127, -128];
        let mut output = [0i8; 3];
        sigmoid_s8(&input, &mut output);
        // Sigmoid(0) should be around 0 in s8 centered representation
        assert!((output[0] as i32).abs() <= 5);
        assert!(output[1] > 100);
        assert!(output[2] < -100);
    }
}
