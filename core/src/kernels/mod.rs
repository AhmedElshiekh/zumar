#![allow(dead_code)]

use candle_core::{Tensor, Result};

/// ضرب مصفوفات BitNet: x @ W^T حيث W هي أوزان 2-bit
/// 
/// المدخلات:
///   x: [M, K] - مصفوفة المدخلات (FP32)
///   packed_w: بايتات الأوزان المضغوطة (2-bit per weight)
///   scale: معامل القياس
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
            "Shape mismatch: x has {} cols, weight has {} cols", k, k_w
        )));
    }
    
    let device = x.device();
    let x_data = x.to_vec2::<f32>()?;  // [M, K]
    
    // فك ضغط الأوزان إلى قيم {-1, 0, +1} مباشرة
    let map = [0.0f32, 0.0f32, 1.0f32, -1.0f32]; // 00→0, 01→0, 10→+1, 11→-1
    
    let mut result = vec![0.0f32; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k_idx in 0..k {
                // استخراج الوزن 2-bit
                let weight_idx = j * k + k_idx;
                let byte_idx = weight_idx / 4;
                let bit_offset = (weight_idx % 4) * 2;
                
                if byte_idx < packed_w.len() {
                    let byte = packed_w[byte_idx];
                    let bits = (byte >> bit_offset) & 0b11;
                    let w_val = map[bits as usize];
                    
                    sum += x_data[i][k_idx] * w_val;
                }
            }
            result[i * n + j] = sum * scale;
        }
    }
    
    Tensor::from_vec(result, (m, n), device)
}

/// إصدار أسرع: معالجة كل 4 أوزان معاً
pub fn bitnet_matmul_fast(
    x: &Tensor,
    packed_w: &[u8],
    scale: f32,
    weight_shape: (usize, usize),
) -> Result<Tensor> {
    let (m, k) = x.dims2()?;
    let (n, k_w) = weight_shape;
    
    if k != k_w {
        return Err(candle_core::Error::Msg("Shape mismatch".to_string()));
    }
    
    let device = x.device();
    let x_data = x.to_vec2::<f32>()?;
    
    // جدول بحث سريع: كل بايت = 4 أوزان
    // byte → [w0, w1, w2, w3] حيث كل وزن = -1, 0, أو +1
    let mut result = vec![0.0f32; m * n];
    
    for i in 0..m {
        let x_row = &x_data[i];
        let res_row = &mut result[i * n..(i + 1) * n];
        
        for j in 0..n {
            let mut sum = 0.0f32;
            let base_idx = j * k;
            
            // معالجة 4 أوزان في كل دورة
            let num_full_chunks = k / 4;
            for chunk in 0..num_full_chunks {
                let byte_idx = (base_idx / 4) + chunk;
                if byte_idx >= packed_w.len() { break; }
                
                let byte = packed_w[byte_idx];
                let x_base = chunk * 4;
                
                // فك تشفير 4 أوزان من بايت واحد
                for bit in 0..4 {
                    let bits = (byte >> (bit * 2)) & 0b11;
                    let w_val = match bits {
                        0b00 => 0.0f32,   // 0
                        0b01 => 0.0f32,   // 0 (احتياطي)
                        0b10 => 1.0f32,   // +1
                        0b11 => -1.0f32,  // -1
                        _ => 0.0f32,
                    };
                    sum += x_row[x_base + bit] * w_val;
                }
            }
            
            // معالجة الباقي (أقل من 4)
            let remainder_start = num_full_chunks * 4;
            for k_idx in remainder_start..k {
                let weight_idx = base_idx + k_idx;
                let byte_idx = weight_idx / 4;
                let bit_offset = (weight_idx % 4) * 2;
                
                if byte_idx < packed_w.len() {
                    let bits = (packed_w[byte_idx] >> bit_offset) & 0b11;
                    let w_val = match bits {
                        0b10 => 1.0f32,
                        0b11 => -1.0f32,
                        _ => 0.0f32,
                    };
                    sum += x_row[k_idx] * w_val;
                }
            }
            
            res_row[j] = sum * scale;
        }
    }
    
    Tensor::from_vec(result, (m, n), device)
}