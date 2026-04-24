use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use crate::kernels;

pub struct ZumarBitLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    scale: Tensor,
}

impl ZumarBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        let raw_weight = vs.get((out_dim, in_dim), "weight")?;
        let device = vs.device();
        let mean_abs = raw_weight.abs()?.mean_all()?.to_scalar::<f32>()?;
        let scale_val = mean_abs.max(1e-5);
        let scale_tensor = Tensor::new(scale_val, device)?;
        let quantized_weight = (raw_weight.broadcast_div(&scale_tensor)?.round()?.clamp(-1.0, 1.0))?;
        Ok(Self { weight: quantized_weight, bias: None, scale: scale_tensor })
    }
}

impl Module for ZumarBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // حفظ الشكل الأصلي
        let original_shape = x.shape().clone();
        
        // تحويل إلى ثنائي الأبعاد للمضاعفة
        let x_2d = if original_shape.rank() == 3 {
            let (b, s, h) = x.dims3()?;
            x.reshape((b * s, h))?
        } else {
            x.clone()
        };

        // الضرب
        let res = if x_2d.device().is_cuda() {
            kernels::bitnet_matmul(&x_2d, &self.weight)?
        } else {
            x_2d.matmul(&self.weight.t()?)?
        };

        // scale
        let res = res.broadcast_mul(&self.scale)?;

        // bias
        let res = match &self.bias {
            Some(b) => res.broadcast_add(b),
            None => Ok(res),
        }?;

        // إعادة التشكيل
        if original_shape.rank() == 3 {
            let (b, s, _) = original_shape.dims3()?;
            let out_dim = self.weight.dim(0)?;
            res.reshape((b, s, out_dim))
        } else {
            Ok(res)
        }
    }
}