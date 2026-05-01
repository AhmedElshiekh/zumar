#![allow(dead_code)]

use candle_core::{Tensor, Result};

// ✅ جدول موحد لفك الضغط — متوافق مع bitlinear.rs:
//   0b00 → 0.0   (صفر)
//   0b01 → 0.0   (غير مستخدم)
//   0b10 → +1.0  (موجب)
//   0b11 → -1.0  (سالب)
// هذا يتطابق مع التصدير في bitlinear.rs:
//   val >= 0.5  → 0b10
//   val <= -0.5 → 0b11
//   else        → 0b00
const DECODE_MAP: [f32; 4] = [0.0f32, 0.0f32, 1.0f32, -1.0f32];

#[inline(always)]
fn decode_2bit(bits: u8) -> f32 {
    DECODE_MAP[(bits & 0b11) as usize]
}

/// ضرب مصفوفات BitNet: x @ W^T حيث W هي أوزان 2-bit
///
/// المدخلات:
///   x:            [M, K] - مصفوفة المدخلات (FP32)
///   packed_w:     بايتات الأوزان المضغوطة (4 أوزان 2-bit لكل بايت)
///   scale:        معامل القياس
///   weight_shape: (N, K) أبعاد الوزن الأصلي
///
/// المخرجات:
///   [M, N] - نتيجة الضرب
pub fn bitnet_matmul(
    x: &Tensor,
    packed_w: &[u8],
    scale: f32,
    weight_shape: (usize, usize),
) -> Result<Tensor> {
    let (m, k) = x.dims2()?;
    let (n, k_w) = weight_shape;

    if k != k_w {
        return Err(candle_core::Error::Msg(format!(
            "Shape mismatch: x=[{},{}] weight=[{},{}]", m, k, n, k_w
        )));
    }

    let device = x.device();
    let x_data = x.to_vec2::<f32>()?;
    let mut result = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k_idx in 0..k {
                // ✅ حساب byte_idx الصحيح — لا يفترض أن k مضاعف لـ 4
                let weight_idx = j * k + k_idx;
                let byte_idx   = weight_idx / 4;
                let bit_offset = (weight_idx % 4) * 2;

                if byte_idx < packed_w.len() {
                    let bits  = (packed_w[byte_idx] >> bit_offset) & 0b11;
                    sum += x_data[i][k_idx] * decode_2bit(bits);
                }
            }
            result[i * n + j] = sum * scale;
        }
    }

    Tensor::from_vec(result, (m, n), device)
}

/// إصدار محسّن: معالجة 4 أوزان في كل دورة
pub fn bitnet_matmul_fast(
    x: &Tensor,
    packed_w: &[u8],
    scale: f32,
    weight_shape: (usize, usize),
) -> Result<Tensor> {
    let (m, k) = x.dims2()?;
    let (n, k_w) = weight_shape;

    if k != k_w {
        return Err(candle_core::Error::Msg(format!(
            "Shape mismatch: x=[{},{}] weight=[{},{}]", m, k, n, k_w
        )));
    }

    let device = x.device();
    let x_data = x.to_vec2::<f32>()?;
    let mut result = vec![0.0f32; m * n];

    for i in 0..m {
        let x_row   = &x_data[i];
        let res_row = &mut result[i * n..(i + 1) * n];

        for j in 0..n {
            let mut sum = 0.0f32;

            // ✅ num_full_chunks بناءً على k الفعلي وليس افتراض التوافق
            let num_full_chunks = k / 4;

            for chunk in 0..num_full_chunks {
                // ✅ byte_idx الصحيح: (j*k + chunk*4) / 4
                let weight_start = j * k + chunk * 4;
                let byte_idx     = weight_start / 4;

                if byte_idx >= packed_w.len() { break; }

                let byte   = packed_w[byte_idx];
                let x_base = chunk * 4;

                // فك تشفير 4 أوزان من بايت واحد — نفس DECODE_MAP
                sum += x_row[x_base]     * decode_2bit(byte);
                sum += x_row[x_base + 1] * decode_2bit(byte >> 2);
                sum += x_row[x_base + 2] * decode_2bit(byte >> 4);
                sum += x_row[x_base + 3] * decode_2bit(byte >> 6);
            }

            // ✅ معالجة الباقي (k ليس مضاعفاً لـ 4)
            let remainder_start = num_full_chunks * 4;
            for k_idx in remainder_start..k {
                let weight_idx = j * k + k_idx;
                let byte_idx   = weight_idx / 4;
                let bit_offset = (weight_idx % 4) * 2;

                if byte_idx < packed_w.len() {
                    let bits = (packed_w[byte_idx] >> bit_offset) & 0b11;
                    sum += x_row[k_idx] * decode_2bit(bits);
                }
            }

            res_row[j] = sum * scale;
        }
    }

    Tensor::from_vec(result, (m, n), device)
}
