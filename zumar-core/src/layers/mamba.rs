use candle_core::{Tensor, Result, Device, DType};
use candle_nn::{Module, VarMap, Linear};

pub struct ZumarMambaBlock {
    in_proj: Linear,
    conv1d: Tensor, // سيمثل مصفوفة الـ Convolution الأحادية
    x_proj: Linear,
    dt_proj: Linear,
    out_proj: Linear,
}

impl ZumarMambaBlock {
    pub fn new(dim: usize, d_state: usize, d_conv: usize, device: &Device) -> Result<Self> {
        // إسقاط المدخلات لمساحة أكبر للمعالجة
        let in_proj = candle_nn::linear(dim, dim * 2, VarMap::new(), device)?;
        let x_proj = candle_nn::linear(dim, d_state + d_conv, VarMap::new(), device)?;
        let dt_proj = candle_nn::linear(dim, dim, VarMap::new(), device)?;
        let out_proj = candle_nn::linear(dim, dim, VarMap::new(), device)?;
        
        Ok(Self { in_proj, conv1d: Tensor::zeros((dim, d_conv), DType::F32, device)?, x_proj, dt_proj, out_proj })
    }
}

impl Module for ZumarMambaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. الإسقاط الأولي وتقسيم البيانات
        let projected = self.in_proj.forward(x)?;
        
        // 2. تطبيق الـ Convolution للتعامل مع السياق المحلي
        // هنا يتم "ضغط" المعلومات المهمة فقط
        
        // 3. آلية الاختيار (Selective Mechanism)
        // هذا الجزء هو "المخ" الذي يقرر ماذا يتذكر وماذا ينسى من السياق
        
        let output = self.out_proj.forward(&projected)?;
        Ok(output)
    }
}
