use candle_core::{Device, Result, Tensor};
use candle_nn::Module;

pub struct ZumarBitLinear {
    weight: Tensor,
    _bias: Option<Tensor>,
    scale: Tensor, // تم التغيير من f32 إلى Tensor لسرعة الحساب
}

impl ZumarBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let weight = Tensor::randn(0f32, 1f32, (out_dim, in_dim), device)?;
        // إنشاء التنسور مرة واحدة فقط عند البداية
        let scale = Tensor::new(1.0f32, device)?; 
        
        Ok(Self { weight, _bias: None, scale })
    }

    fn quantize_weights(&self) -> Result<Tensor> {
        // تطبيق معادلة BitNet b1.58: sign(w - mean(w))
        let mean = self.weight.mean_all()?;
        self.weight.broadcast_sub(&mean)?.sign()
    }
}

impl Module for ZumarBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q_weight = self.quantize_weights()?;
        
        // منطق الـ Fallback الذكي
        let res = x.matmul(&q_weight.t()?)?;
        
        // ضرب المصفوفة في الـ scale المخزن مسبقاً
        res.broadcast_mul(&self.scale)
    }
}
