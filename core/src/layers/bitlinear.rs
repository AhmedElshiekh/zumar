use candle_core::{Tensor, Result, DType};
use candle_nn::{Module, VarBuilder};

/// طبقة 1-bit خطية مع دعم التدريب والتكميم
pub struct ZumarBitLinear {
    /// الأوزان المخزنة بدقة FP32 للتدريب
    pub latent_weight: Tensor,
    /// الأوزان المكممة (للأستدلال فقط)
    pub quantized_weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    /// معامل القياس
    pub scale: Tensor,
    /// هل نستخدم التكميم؟
    pub quantize: bool,
}

impl ZumarBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        // تهيئة أوزان عشوائية
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
            quantize: false, // يبدأ بدون تكميم للتدريب
        })
    }
    
    /// تكميم الأوزان إلى {-1, 0, 1}
    pub fn quantize_weights(&mut self) -> Result<()> {
        let mean_abs = self.latent_weight.abs()?
            .mean_all()?
            .to_scalar::<f32>()?
            .max(1e-6);
        
        self.scale = Tensor::new(mean_abs, self.latent_weight.device())?;
        
        // BitNet b1.58: وزن لكل من القيم الثلاثة
        let scaled = self.latent_weight.broadcast_div(&self.scale)?;
        let rounded = scaled.round()?;
        let clamped = rounded.clamp(-1.0f64, 1.0f64)?;
        
        self.quantized_weight = Some(clamped);
        self.quantize = true;
        
        println!("   📊 Quantized weights: scale={:.6}", mean_abs);
        Ok(())
    }
    
    /// حساب الضرب المصفوفي مع الأوزان المكممة
    fn quantized_matmul(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(ref qw) = self.quantized_weight {
            // استخدام الضرب المحسن لـ 1-bit
            // بدلاً من x @ W^T، نستخدم عمليات أسرع
            let x_f32 = x.to_dtype(DType::F32)?;
            let qw_f32 = qw.to_dtype(DType::F32)?;
            
            // الضرب العادي لكن الأوزان هي {-1, 0, 1}
            let result = x_f32.matmul(&qw_f32.t()?)?;
            
            // تطبيق scale
            result.broadcast_mul(&self.scale)
        } else {
            // رجوع للضرب العادي
            x.matmul(&self.latent_weight.t()?)
        }
    }
}

impl Module for ZumarBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_shape = x.shape().clone();
        
        // تحويل إلى شكل ثنائي الأبعاد إذا لزم
        let x_2d = if original_shape.rank() == 3 {
            let (b, s, h) = x.dims3()?;
            x.reshape((b * s, h))?
        } else {
            x.clone()
        };
        
        // اختر بين الضرب المكمم والعادي
        let res = if self.quantize && self.quantized_weight.is_some() {
            self.quantized_matmul(&x_2d)?
        } else {
            x_2d.matmul(&self.latent_weight.t()?)?
        };
        
        // تطبيق bias إذا وجد
        let res = match &self.bias {
            Some(b) => res.broadcast_add(b),
            None => Ok(res),
        }?;
        
        // إعادة التشكيل
        if original_shape.rank() == 3 {
            let (b, s, _) = original_shape.dims3()?;
            let out_dim = self.latent_weight.dim(0)?;
            res.reshape((b, s, out_dim))
        } else {
            Ok(res)
        }
    }
}