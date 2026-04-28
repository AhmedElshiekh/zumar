use candle_core::{Tensor, Result, DType};
use candle_nn::{Module, VarBuilder};
use crate::kernels;

pub struct ZumarBitLinear {
    pub latent_weight: Tensor,
    pub quantized_weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub scale: Tensor,
    pub quantize: bool,
    /// الأوزان المضغوطة 2-bit (للـ kernel المباشر)
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
    
    /// تحميل من بيانات .zmr المضغوطة
    pub fn from_zmr(data: &[u8], shape: (usize, usize), device: &candle_core::Device) -> Result<Self> {
        if data.len() < 4 {
            return Err(candle_core::Error::Msg("Invalid packed data".to_string()));
        }
        
        let scale_val = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let scale = Tensor::new(scale_val, device)?;
        let packed = data[4..].to_vec();
        
        // وزن وهمي للتوافق
        let latent = Tensor::zeros(shape, DType::F32, device)?;
        
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
    
    /// استخدام BitNet kernel مباشرة
    pub fn forward_bitnet(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(ref packed) = self.packed_2bit {
            let scale_val = self.scale.to_scalar::<f32>()?;
            kernels::bitnet_matmul_fast(x, packed, scale_val, self.weight_shape)
        } else {
            // رجوع للطريقة العادية
            x.matmul(&self.latent_weight.t()?)
        }
    }
    
    /// تصدير الأوزان إلى 2-bit
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
                let bits: u8 = if val <= -0.5 {
                    0b11  // -1
                } else if val >= 0.5 {
                    0b10  // +1
                } else {
                    0b00  // 0
                };
                byte |= bits << (i * 2);
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
        
        // استخدام BitNet kernel إذا كانت الأوزان المضغوطة متاحة
        let res = if self.packed_2bit.is_some() {
            self.forward_bitnet(&x_2d)?
        } else if let Some(ref qw) = self.quantized_weight {
            x_2d.matmul(&qw.to_dtype(DType::F32)?.t()?)?.broadcast_mul(&self.scale)?
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