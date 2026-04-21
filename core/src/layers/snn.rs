pub struct ZumarSpikingLayer {
    threshold: f16,
    v_mem: Tensor, // جهد الغشاء (Membrane Potential)
}

impl ZumarSpikingLayer {
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // إضافة الإشارة القادمة إلى الجهد الحالي
        self.v_mem = (&self.v_mem + x)?;
        
        // إذا تجاوز الجهد العتبة، أرسل "نبضة" (1)، وإلا (0)
        let mask = self.v_mem.gt(self.threshold)?;
        let spikes = mask.to_dtype(DType::F16)?;
        
        // تصفير الجهد للأماكن التي أرسلت نبضات (Reset)
        self.v_mem = self.v_mem.where_cond(&mask, &Tensor::zeros_like(&self.v_mem)?)?;
        
        Ok(spikes)
    }
}
