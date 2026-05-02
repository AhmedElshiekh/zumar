use candle_core::{Device, Result, Tensor, DType};

/// ✅ Packing حقيقي لأوزان 2-bit (BitNet b1.58)
///
/// ترميز القيم:
///   0.0  → 0b00
///  +1.0  → 0b10
///  -1.0  → 0b11
///
/// كل بايت يحتوي 4 أوزان — ضغط 16× مقارنة بـ f32
pub struct BitPacker;

impl BitPacker {
    /// تحويل tensor f32 إلى بايتات مضغوطة 2-bit
    /// المدخل: tensor بقيم {-1.0, 0.0, +1.0}
    /// المخرج: Vec<u8> حيث كل بايت يحمل 4 أوزان
    pub fn pack(tensor: &Tensor) -> Result<(Vec<u8>, f32, Vec<usize>)> {
        let shape  = tensor.dims().to_vec();
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        let total  = values.len();

        // ── quantize إلى {-1, 0, +1} مع حفظ الـ scale ──────────
        let abs_mean: f32 = values.iter().map(|v| v.abs()).sum::<f32>()
            / total.max(1) as f32;
        let scale = abs_mean.max(1e-8);

        let mut packed = Vec::with_capacity((total + 3) / 4);
        let mut byte   = 0u8;
        let mut count  = 0usize;

        for &val in &values {
            let normalized = val / scale;
            let bits: u8 = if normalized >= 0.5 {
                0b10   // +1
            } else if normalized <= -0.5 {
                0b11   // -1
            } else {
                0b00   //  0
            };

            // وضع الـ 2-bit في موضعها من البايت
            let offset = (count % 4) * 2;
            byte |= bits << offset;
            count += 1;

            if count % 4 == 0 {
                packed.push(byte);
                byte = 0;
            }
        }
        // آخر بايت جزئي
        if count % 4 != 0 {
            packed.push(byte);
        }

        Ok((packed, scale, shape))
    }

    /// ✅ فك الضغط: بايتات → tensor f32
    pub fn unpack(
        packed: &[u8],
        scale:  f32,
        shape:  &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let total   = shape.iter().product::<usize>();
        // نفس DECODE_MAP الموحد من kernels/mod.rs
        const MAP: [f32; 4] = [0.0f32, 0.0f32, 1.0f32, -1.0f32];

        let mut values = Vec::with_capacity(total);

        'outer: for &byte in packed {
            for bit in 0..4 {
                if values.len() >= total { break 'outer; }
                let bits = (byte >> (bit * 2)) & 0b11;
                values.push(MAP[bits as usize] * scale);
            }
        }
        values.truncate(total);

        Tensor::from_vec(values, shape, device)
    }

    /// واجهة مبسطة: pack tensor مباشرة (للتوافق مع loader.rs)
    pub fn pack_weights(tensor: &Tensor) -> Result<Tensor> {
        let (packed_bytes, _scale, _shape) = Self::pack(tensor)?;
        Tensor::from_vec(packed_bytes, packed_bytes.len(), tensor.device())
    }

    /// حجم البايتات المتوقع بعد الضغط
    pub fn packed_size(num_weights: usize) -> usize {
        (num_weights + 3) / 4
    }

    /// نسبة الضغط مقارنة بـ f32
    pub fn compression_ratio() -> f32 {
        16.0  // f32=32bit, 2bit → نسبة 16×
    }
}

/// واجهة Tensor مباشرة للـ pack/unpack
pub trait PackableTensor {
    fn pack_2bit(&self) -> Result<(Vec<u8>, f32, Vec<usize>)>;
}

impl PackableTensor for Tensor {
    fn pack_2bit(&self) -> Result<(Vec<u8>, f32, Vec<usize>)> {
        BitPacker::pack(self)
    }
}
