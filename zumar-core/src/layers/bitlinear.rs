use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct ZumarBitLinear {
    weight: Tensor,
    _bias: Option<Tensor>,
    scale: Tensor, 
}

impl ZumarBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        let raw_weight = vs.get((out_dim, in_dim), "weight")?;
        let device = vs.device();

        let mean_abs = raw_weight.abs()?.mean_all()?.to_scalar::<f32>()?;
        let scale_val = mean_abs.max(1e-5); 
        let scale_tensor = Tensor::new(scale_val, device)?;

        let quantized_weight = (raw_weight.broadcast_div(&scale_tensor)?
            .round()?
            .clamp(-1.0, 1.0))?;

        Ok(Self { 
            weight: quantized_weight, 
            _bias: None, 
            scale: scale_tensor 
        })
    }

    pub fn _update_quantization(&mut self, new_weight: Tensor) -> Result<()> {
        let mean_abs = new_weight.abs()?.mean_all()?.to_scalar::<f32>()?;
        self.scale = Tensor::new(mean_abs.max(1e-5), new_weight.device())?;
        self.weight = (new_weight.broadcast_div(&self.scale)?
            .round()?
            .clamp(-1.0, 1.0))?;
        Ok(())
    }
}

impl Module for ZumarBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [Batch, Seq, Hidden] -> [1, 10, 768]
        // نحن بحاجة لتحويله إلى [Batch * Seq, Hidden] -> [10, 768]
        let shape = x.shape();
        let dims = shape.dims();
        
        // استخراج الأبعاد ديناميكياً
        let (b, s, h) = (dims[0], dims[1], dims[2]);

        // 1. تسطيح الأبعاد الأولى لضمان توافق Matmul
        let x_flat = x.reshape((b * s, h))?;
        
        // 2. تنفيذ الضرب: [10, 768] * [768, Out] = [10, Out]
        let w_t = self.weight.t()?;
        let res_flat = x_flat.matmul(&w_t)?;
        
        // 3. إعادة التنسور لأبعاده الأصلية: [1, 10, Out]
        let out_dim = w_t.dim(1)?;
        let res = res_flat.reshape((b, s, out_dim))?;
        
        // 4. تطبيق الـ Scale
        res.broadcast_mul(&self.scale)
    }
}
