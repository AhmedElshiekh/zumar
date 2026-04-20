//use candle_core::{Result, Tensor, Device, Shape};
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use crate::kernels; // استدعاء الموديول الخاص بالكيرنالات

pub struct ZumarBitLinear {
    weight: Tensor, // مخزنة كـ {-1, 0, 1} ولكن بنوع f32 حالياً
    bias: Option<Tensor>,
    scale: Tensor,
}

impl ZumarBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        let raw_weight = vs.get((out_dim, in_dim), "weight")?;
        let device = vs.device();

        // حساب الـ Scale بناءً على متوسط القيم
        let mean_abs = raw_weight.abs()?.mean_all()?.to_scalar::<f32>()?;
        let scale_val = mean_abs.max(1e-5);
        let scale_tensor = Tensor::new(scale_val, device)?;

        // تكميم الأوزان إلى {-1, 0, 1}
        let quantized_weight = (raw_weight.broadcast_div(&scale_tensor)?
            .round()?
            .clamp(-1.0, 1.0))?;

        Ok(Self {
            weight: quantized_weight,
            bias: None,
            scale: scale_tensor,
        })
    }
}

impl Module for ZumarBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;

        // --- [ نقطة التحول للسرعة: استدعاء الكيرنال ] ---
        
        let res = if x.device().is_cuda() {
            // إذا كان الجهاز CUDA، نستخدم الكيرنال المخصص الذي يعالج العمليات كـ Integers
            // هذا سيعوض x_flat.matmul(&self.weight.t()?)
            kernels::bitnet_matmul(&x_flat, &self.weight)?
        } else {
            // إذا كان على الـ CPU، نستخدم Matmul المحسن للـ Native CPU
            x_flat.matmul(&self.weight.t()?)?
        };

        // إعادة التشكيل وتطبيق الـ Scale
        let out_dim = self.weight.dim(0)?;
        let res = res.reshape((b, s, out_dim))?;
        
        let res = res.broadcast_mul(&self.scale)?;

        match &self.bias {
            Some(b) => res.broadcast_add(b),
            None => Ok(res),
        }
    }
}
