use candle_core::{Tensor, Result};
use crate::layers::snn::ZumarSpikingLayer;
use crate::layers::bitlinear::ZumarBitLinear;

pub struct ZumarBlock {
    linear: ZumarBitLinear,
    snn: ZumarSpikingLayer,
}

impl ZumarBlock {
    pub fn new(in_dim: usize, out_dim: usize, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            linear: ZumarBitLinear::new(in_dim, out_dim, device)?,
            // إعداد العتبة (Threshold)؛ إذا تجاوزت الإشارة 1.0 يتم إرسال نبضة
            snn: ZumarSpikingLayer::new(1.0, out_dim, device)?, 
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // 1. تمرير البيانات عبر طبقة الـ 1-bit
        let x = self.linear.forward(x)?;
        
        // 2. تصفية الإشارة عبر النبضات العصبية
        // هذا يقلل من الضجيج ويوفر الطاقة لأن القيم الصفرية لا تستهلك حسابات لاحقاً
        let x = self.snn.forward(&x)?;
        
        Ok(x)
    }
}
