use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarMap};

/// طبقة الخطية لـ Zumar (1-Bit Linear Layer)
/// تعتمد فلسفة تقليل الحسابات عبر تكميم الأوزان لحظياً
pub struct ZumarBitLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    scale: f32,
}

impl ZumarBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // إنشاء أوزان عشوائية مبدئياً (في الواقع سيتم تحميلها من ملف safetensors)
        let weight = Tensor::randn(0f32, 1f32, (out_dim, in_dim), device)?;
        
        Ok(Self {
            weight,
            bias: None,
            scale: 1.0, // سيتم حسابه بناءً على توزيع الأوزان
        })
    }

    /// عملية التكميم (Quantization) للأوزان إلى {-1, 0, 1}
    fn quantize_weights(&self) -> Result<Tensor> {
        // حساب المتوسط لضبط المقياس (Scaling Factor)
        let mean = self.weight.mean_all()?;
        let centered = self.weight.broadcast_sub(&mean)?;
        
        // التكميم: تحويل القيم إلى -1 أو 1 بناءً على الإشارة (Sign)
        // ملاحظة: في BitNet b1.58 نستخدم عتبة للصفر أيضاً
        let quantized = centered.sign()?;
        Ok(quantized)
    }
}

impl Module for ZumarBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q_weight = self.quantize_weights()?;
        
        // هنا تكمن السحر: matmul مع أوزان 1-bit يتحول برمجياً 
        // إلى عمليات Addition/Subtraction فقط داخل الـ Kernel
        let output = x.matmul(&q_weight.t()?)?;
        
        // ضرب الناتج في مقياس (Scale) لاستعادة النطاق الديناميكي
        output.mul(self.scale)
    }
}
