use candle_core::{Tensor, Result, DType};
use candle_nn::{Module, VarBuilder};

pub struct ZumarBitLinear {
    pub latent_weight: Tensor,
    pub quantized_weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub scale: Tensor,
    pub quantize: bool,
    /// الأوزان بصيغة 2-bit مضغوطة (للتحميل المباشر)
    pub packed_2bit: Option<Vec<u8>>,
    pub weight_shape: (usize, usize),
}

impl ZumarBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        let latent_weight = vs.get_with_hints(
            (out_dim, in_dim), 
            "weight",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 }
        )?;
        
        let device = vs.device();
        let scale = Tensor::new(1.0f32, device)?;
        
        Ok(Self {
            latent_weight,
            quantized_weight: None,
            bias: None,
            scale,
            quantize: true,
            packed_2bit: None,
            weight_shape: (out_dim, in_dim),
        })
    }
    
    /// إنشاء من بيانات .zmr المضغوطة مباشرة
    pub fn from_zmr(data: &[u8], shape: (usize, usize), device: &candle_core::Device) -> Result<Self> {
        if data.len() < 4 {
            return Err(candle_core::Error::Msg("Invalid packed data".to_string()));
        }
        
        // قراءة scale
        let scale_val = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let scale = Tensor::new(scale_val, device)?;
        
        // الأوزان المضغوطة (2-bit)
        let packed = data[4..].to_vec();
        
        // فك ضغط الأوزان إلى FP32 للاستخدام
        let total = shape.0 * shape.1;
        let mut weights = Vec::with_capacity(total);
        let map = [-1.0f32, 0.0f32, 1.0f32, 0.0f32]; // 00→-1, 01→0, 10→+1, 11→0
        
        for &byte in &packed {
            for i in 0..4 {
                if weights.len() >= total { break; }
                let bits = (byte >> (i * 2)) & 0b11;
                weights.push(map[bits as usize] * scale_val);
            }
        }
        weights.truncate(total);
        
        let latent = Tensor::from_vec(weights, shape, device)?;
        
        Ok(Self {
            latent_weight: latent.clone(),
            quantized_weight: Some(latent),
            bias: None,
            scale,
            quantize: true,
            packed_2bit: Some(packed),
            weight_shape: shape,
        })
    }
    
    pub fn quantize_weights(&mut self) -> Result<()> {
        let mean_abs = self.latent_weight.abs()?
            .mean_all()?
            .to_scalar::<f32>()?
            .max(1e-6);
        
        self.scale = Tensor::new(mean_abs, self.latent_weight.device())?;
        
        let scaled = self.latent_weight.broadcast_div(&self.scale)?;
        let rounded = scaled.round()?;
        let clamped = rounded.clamp(-1.0f64, 1.0f64)?;
        
        self.quantized_weight = Some(clamped);
        self.quantize = true;
        
        Ok(())
    }
    
    /// تصدير الأوزان كـ 2-bit للتخزين المضغوط
    pub fn to_packed_bytes(&self) -> Result<Vec<u8>> {
        let weight = if let Some(ref qw) = self.quantized_weight {
            qw.to_dtype(DType::F32)?
        } else {
            self.latent_weight.to_dtype(DType::F32)?
        };
        
        let scale = self.scale.to_scalar::<f32>()?;
        let flat = weight.flatten_all()?.to_vec1::<f32>()?;
        
        let mut packed = Vec::new();
        packed.extend_from_slice(&scale.to_le_bytes());
        
        for chunk in flat.chunks(4) {
            let mut byte: u8 = 0;
            for (i, &val) in chunk.iter().enumerate() {
                let quantized: u8 = if val <= -0.5 {
                    0b00
                } else if val >= 0.5 {
                    0b10
                } else {
                    0b01
                };
                byte |= quantized << (i * 2);
            }
            packed.push(byte);
        }
        
        Ok(packed)
    }
}

impl Module for ZumarBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_shape = x.shape().clone();
        
        let x_2d = if original_shape.rank() == 3 {
            let (b, s, h) = x.dims3()?;
            x.reshape((b * s, h))?
        } else {
            x.clone()
        };
        
        let res = if let Some(ref qw) = self.quantized_weight {
            let qw_f32 = qw.to_dtype(DType::F32)?;
            x_2d.matmul(&qw_f32.t()?)?.broadcast_mul(&self.scale)?
        } else {
            x_2d.matmul(&self.latent_weight.t()?)?
        };
        
        let res = match &self.bias {
            Some(b) => res.broadcast_add(b),
            None => Ok(res),
        }?;
        
        if original_shape.rank() == 3 {
            let (b, s, _) = original_shape.dims3()?;
            let out_dim = self.latent_weight.dim(0)?;
            res.reshape((b, s, out_dim))
        } else {
            Ok(res)
        }
    }
}