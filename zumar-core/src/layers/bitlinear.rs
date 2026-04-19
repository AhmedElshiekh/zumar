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
        
        // استخدام الـ Custom Kernel بدلاً من x.matmul
        #[cfg(feature = "cuda")]
        {
            if x.device().is_cuda() {
                let out_dim = q_weight.dims()[0];
                let in_dim = x.dims()[0];
                return crate::kernels::launch_bitnet_kernel(x, &q_weight, (in_dim, out_dim));
            }
        }

        // كود احتياطي (Fallback) للـ CPU إذا لم يتوفر CUDA
        x.matmul(&q_weight.t()?)?.mul(self.scale)
    }
}
