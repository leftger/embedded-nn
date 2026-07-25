//! Advanced recurrent neural network layers (LSTM, SVDF).

use crate::activations::{sigmoid_s16, tanh_s16};
use crate::simd::{vec_dot_s16, vec_dot_s8};
use crate::support::{clamp, requantize};
use crate::types::{Activation, PerTensorQuantParams, Result};

/// Quantized parameters for an LSTM cell gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LstmGateParams {
    /// Zero point offset for input.
    pub input_offset: i32,
    /// Zero point offset for recurrent hidden state.
    pub hidden_offset: i32,
    /// Requantization multiplier for gate pre-activation.
    pub multiplier: i32,
    /// Requantization shift for gate pre-activation.
    pub shift: i32,
}

/// Unidirectional LSTM cell step for quantized int8/int16 state tensors (`lstm_step_s8_s16`).
pub fn lstm_step_s8_s16(
    input: &[i8],
    hidden_state: &mut [i8],
    cell_state: &mut [i16],
    weight_input: &[i8],   // 4 * hidden_dim x input_dim (i, f, g, o)
    weight_hidden: &[i8],  // 4 * hidden_dim x hidden_dim (i, f, g, o)
    bias: &[i32],          // 4 * hidden_dim
    gate_params: &LstmGateParams,
    cell_clip: i16,
    output_quant: &PerTensorQuantParams,
    output_offset: i32,
    activation: &Activation,
) -> Result<()> {
    let input_dim = input.len();
    let hidden_dim = hidden_state.len();

    for h in 0..hidden_dim {
        // Gates layout: 0: input (i), 1: forget (f), 2: cell (g), 3: output (o)
        let w_i_in = &weight_input[0 * hidden_dim * input_dim + h * input_dim..];
        let w_f_in = &weight_input[1 * hidden_dim * input_dim + h * input_dim..];
        let w_g_in = &weight_input[2 * hidden_dim * input_dim + h * input_dim..];
        let w_o_in = &weight_input[3 * hidden_dim * input_dim + h * input_dim..];

        let w_i_h = &weight_hidden[0 * hidden_dim * hidden_dim + h * hidden_dim..];
        let w_f_h = &weight_hidden[1 * hidden_dim * hidden_dim + h * hidden_dim..];
        let w_g_h = &weight_hidden[2 * hidden_dim * hidden_dim + h * hidden_dim..];
        let w_o_h = &weight_hidden[3 * hidden_dim * hidden_dim + h * hidden_dim..];

        // 1. Accumulate pre-activations
        let acc_i = bias[0 * hidden_dim + h]
            + vec_dot_s8(input, w_i_in, gate_params.input_offset)
            + vec_dot_s8(hidden_state, w_i_h, gate_params.hidden_offset);
        let acc_f = bias[1 * hidden_dim + h]
            + vec_dot_s8(input, w_f_in, gate_params.input_offset)
            + vec_dot_s8(hidden_state, w_f_h, gate_params.hidden_offset);
        let acc_g = bias[2 * hidden_dim + h]
            + vec_dot_s8(input, w_g_in, gate_params.input_offset)
            + vec_dot_s8(hidden_state, w_g_h, gate_params.hidden_offset);
        let acc_o = bias[3 * hidden_dim + h]
            + vec_dot_s8(input, w_o_in, gate_params.input_offset)
            + vec_dot_s8(hidden_state, w_o_h, gate_params.hidden_offset);

        // 2. Requantize pre-activations to int16 range
        let req_i = requantize(acc_i, gate_params.multiplier, gate_params.shift) as i16;
        let req_f = requantize(acc_f, gate_params.multiplier, gate_params.shift) as i16;
        let req_g = requantize(acc_g, gate_params.multiplier, gate_params.shift) as i16;
        let req_o = requantize(acc_o, gate_params.multiplier, gate_params.shift) as i16;

        let mut gate_i = [0i16];
        let mut gate_f = [0i16];
        let mut gate_g = [0i16];
        let mut gate_o = [0i16];

        sigmoid_s16(&[req_i], &mut gate_i, 0);
        sigmoid_s16(&[req_f], &mut gate_f, 0);
        tanh_s16(&[req_g], &mut gate_g, 0);
        sigmoid_s16(&[req_o], &mut gate_o, 0);

        // 3. Compute cell state c_t = f * c_prev + i * g
        let c_prev = cell_state[h] as i32;
        let f_val = (gate_f[0] as i32 + 32768) >> 1; // Q15 scale
        let i_val = (gate_i[0] as i32 + 32768) >> 1;
        let g_val = gate_g[0] as i32;

        let c_next = ((f_val * c_prev) >> 15) + ((i_val * g_val) >> 15);
        let c_clamped = clamp(c_next, -cell_clip as i32, cell_clip as i32) as i16;
        cell_state[h] = c_clamped;

        // 4. Compute hidden state h_t = o * tanh(c_t)
        let mut tan_c = [0i16];
        tanh_s16(&[c_clamped], &mut tan_c, 0);

        let o_val = (gate_o[0] as i32 + 32768) >> 1;
        let h_next_raw = (o_val * (tan_c[0] as i32)) >> 15;
        let h_req = requantize(h_next_raw, output_quant.multiplier, output_quant.shift);
        let h_final = clamp(h_req + output_offset, activation.min, activation.max);

        hidden_state[h] = h_final as i8;
    }

    Ok(())
}

/// Full int16 Unidirectional LSTM cell step (`lstm_step_s16`).
pub fn lstm_step_s16(
    input: &[i16],
    hidden_state: &mut [i16],
    cell_state: &mut [i16],
    weight_input: &[i8],
    weight_hidden: &[i8],
    bias: &[i64],
    gate_params: &LstmGateParams,
    cell_clip: i16,
    output_quant: &PerTensorQuantParams,
    activation: &Activation,
) -> Result<()> {
    let input_dim = input.len();
    let hidden_dim = hidden_state.len();

    for h in 0..hidden_dim {
        let w_i_in = &weight_input[0 * hidden_dim * input_dim + h * input_dim..];
        let w_f_in = &weight_input[1 * hidden_dim * input_dim + h * input_dim..];
        let w_g_in = &weight_input[2 * hidden_dim * input_dim + h * input_dim..];
        let w_o_in = &weight_input[3 * hidden_dim * input_dim + h * input_dim..];

        let w_i_h = &weight_hidden[0 * hidden_dim * hidden_dim + h * hidden_dim..];
        let w_f_h = &weight_hidden[1 * hidden_dim * hidden_dim + h * hidden_dim..];
        let w_g_h = &weight_hidden[2 * hidden_dim * hidden_dim + h * hidden_dim..];
        let w_o_h = &weight_hidden[3 * hidden_dim * hidden_dim + h * hidden_dim..];

        let mut acc_i = bias[0 * hidden_dim + h];
        let mut acc_f = bias[1 * hidden_dim + h];
        let mut acc_g = bias[2 * hidden_dim + h];
        let mut acc_o = bias[3 * hidden_dim + h];

        for i in 0..input_dim {
            let in_val = input[i] as i64;
            acc_i += in_val * (w_i_in[i] as i64);
            acc_f += in_val * (w_f_in[i] as i64);
            acc_g += in_val * (w_g_in[i] as i64);
            acc_o += in_val * (w_o_in[i] as i64);
        }

        for i in 0..hidden_dim {
            let h_val = hidden_state[i] as i64;
            acc_i += h_val * (w_i_h[i] as i64);
            acc_f += h_val * (w_f_h[i] as i64);
            acc_g += h_val * (w_g_h[i] as i64);
            acc_o += h_val * (w_o_h[i] as i64);
        }

        let req_i = requantize((acc_i >> 15) as i32, gate_params.multiplier, gate_params.shift) as i16;
        let req_f = requantize((acc_f >> 15) as i32, gate_params.multiplier, gate_params.shift) as i16;
        let req_g = requantize((acc_g >> 15) as i32, gate_params.multiplier, gate_params.shift) as i16;
        let req_o = requantize((acc_o >> 15) as i32, gate_params.multiplier, gate_params.shift) as i16;

        let mut gate_i = [0i16];
        let mut gate_f = [0i16];
        let mut gate_g = [0i16];
        let mut gate_o = [0i16];

        sigmoid_s16(&[req_i], &mut gate_i, 0);
        sigmoid_s16(&[req_f], &mut gate_f, 0);
        tanh_s16(&[req_g], &mut gate_g, 0);
        sigmoid_s16(&[req_o], &mut gate_o, 0);

        let c_prev = cell_state[h] as i32;
        let f_val = (gate_f[0] as i32 + 32768) >> 1;
        let i_val = (gate_i[0] as i32 + 32768) >> 1;
        let g_val = gate_g[0] as i32;

        let c_next = ((f_val * c_prev) >> 15) + ((i_val * g_val) >> 15);
        let c_clamped = clamp(c_next, -cell_clip as i32, cell_clip as i32) as i16;
        cell_state[h] = c_clamped;

        let mut tan_c = [0i16];
        tanh_s16(&[c_clamped], &mut tan_c, 0);

        let o_val = (gate_o[0] as i32 + 32768) >> 1;
        let h_next_raw = (o_val * (tan_c[0] as i32)) >> 15;
        let h_req = requantize(h_next_raw, output_quant.multiplier, output_quant.shift);
        let h_final = clamp(h_req, activation.min, activation.max);

        hidden_state[h] = h_final as i16;
    }

    Ok(())
}

/// SVDF (Singular Value Decomposition Filter) layer step for int8 tensors (`svdf_s8`).
pub fn svdf_s8(
    input_offset: i32,
    output_offset: i32,
    rank: usize,
    input: &[i8],
    state: &mut [i8],
    weights_feature: &[i8],
    weights_time: &[i8],
    bias: Option<&[i32]>,
    input_quant: &PerTensorQuantParams,
    output_quant: &PerTensorQuantParams,
    activation: &Activation,
    output: &mut [i8],
) -> Result<()> {
    let input_dim = input.len();
    let feature_dim = weights_feature.len() / input_dim;
    let time_steps = weights_time.len() / feature_dim;
    let units = feature_dim / rank;

    for f in 0..feature_dim {
        let state_slice = &mut state[f * time_steps..(f + 1) * time_steps];
        state_slice.copy_within(1..time_steps, 0);
    }

    for f in 0..feature_dim {
        let wf = &weights_feature[f * input_dim..(f + 1) * input_dim];
        let acc = vec_dot_s8(input, wf, input_offset);
        let req = requantize(acc, input_quant.multiplier, input_quant.shift);
        let clamped = clamp(req, i8::MIN as i32, i8::MAX as i32);

        state[f * time_steps + (time_steps - 1)] = clamped as i8;
    }

    for u in 0..units {
        let mut acc = match bias {
            Some(b) => b[u],
            None => 0,
        };

        for r in 0..rank {
            let f = u * rank + r;
            let st = &state[f * time_steps..(f + 1) * time_steps];
            let wt = &weights_time[f * time_steps..(f + 1) * time_steps];

            acc += vec_dot_s8(st, wt, 0);
        }

        let req = requantize(acc, output_quant.multiplier, output_quant.shift);
        let final_val = clamp(req + output_offset, activation.min, activation.max);
        output[u] = final_val as i8;
    }

    Ok(())
}

/// SVDF layer step with int16 state tensor (`svdf_state_s16_s8`).
pub fn svdf_state_s16_s8(
    input_offset: i32,
    output_offset: i32,
    rank: usize,
    input: &[i8],
    state: &mut [i16], // 16-bit state tensor for high precision
    weights_feature: &[i8],
    weights_time: &[i16],
    bias: Option<&[i32]>,
    input_quant: &PerTensorQuantParams,
    output_quant: &PerTensorQuantParams,
    activation: &Activation,
    output: &mut [i8],
) -> Result<()> {
    let input_dim = input.len();
    let feature_dim = weights_feature.len() / input_dim;
    let time_steps = weights_time.len() / feature_dim;
    let units = feature_dim / rank;

    for f in 0..feature_dim {
        let state_slice = &mut state[f * time_steps..(f + 1) * time_steps];
        state_slice.copy_within(1..time_steps, 0);
    }

    for f in 0..feature_dim {
        let wf = &weights_feature[f * input_dim..(f + 1) * input_dim];
        let acc = vec_dot_s8(input, wf, input_offset);
        let req = requantize(acc, input_quant.multiplier, input_quant.shift);
        let clamped = clamp(req, i16::MIN as i32, i16::MAX as i32);

        state[f * time_steps + (time_steps - 1)] = clamped as i16;
    }

    for u in 0..units {
        let mut acc = match bias {
            Some(b) => b[u] as i64,
            None => 0i64,
        };

        for r in 0..rank {
            let f = u * rank + r;
            let st = &state[f * time_steps..(f + 1) * time_steps];
            let wt = &weights_time[f * time_steps..(f + 1) * time_steps];

            acc += vec_dot_s16(st, wt);
        }

        let req = requantize((acc >> 15) as i32, output_quant.multiplier, output_quant.shift);
        let final_val = clamp(req + output_offset, activation.min, activation.max);
        output[u] = final_val as i8;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svdf_s8_basic() {
        let rank = 1usize;
        let input = [10i8, 20i8];
        let mut state = [0i8; 2];
        let weights_feature = [1i8, 1i8];
        let weights_time = [1i8, 1i8];
        let bias = [0i32];

        let input_quant = PerTensorQuantParams::new(1073741824, 0);
        let output_quant = PerTensorQuantParams::new(1073741824, 0);
        let act = Activation::int8_unconstrained();
        let mut output = [0i8; 1];

        svdf_s8(
            0,
            0,
            rank,
            &input,
            &mut state,
            &weights_feature,
            &weights_time,
            Some(&bias),
            &input_quant,
            &output_quant,
            &act,
            &mut output,
        )
        .unwrap();

        assert_eq!(output[0], 8);
    }
}
