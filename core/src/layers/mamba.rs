use candle_core::{Tensor, Result, DType}; 
use candle_nn::{Module, VarBuilder, Linear};

pub struct ZumarMambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
}

pub struct ZumarMambaBlock {
    pub in_proj: Linear,
    pub _conv1d: Tensor,
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub out_proj: Linear,
    pub a_log: Tensor,
    pub d: Tensor,
    pub _d_state: usize,
}

impl ZumarMambaBlock {
    pub fn new(cfg: &ZumarMambaConfig, vs: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_model * cfg.expand; 
        let device = vs.device();

        // إسقاط أولي لإنتاج مساري البيانات والـ Gate
        let in_proj = candle_nn::linear(cfg.d_model, d_inner * 2, vs.pp("in_proj"))?;
        let x_proj = candle_nn::linear(d_inner, cfg.d_state * 2 + d_inner, vs.pp("x_proj"))?;
        let dt_proj = candle_nn::linear(d_inner, d_inner, vs.pp("dt_proj"))?;
        let out_proj = candle_nn::linear(d_inner, cfg.d_model, vs.pp("out_proj"))?;

        let a_log = vs.get((cfg.d_state, d_inner), "a_log")?;
        let d = vs.get(d_inner, "d")?;
        let conv1d = Tensor::zeros((d_inner, cfg.d_conv), DType::F16, device)?;

        Ok(Self { 
            in_proj, 
            _conv1d: conv1d, 
            x_proj, 
            dt_proj, 
            out_proj,
            a_log,
            d,
            _d_state: cfg.d_state,
        })
    }

    pub fn apply_ssm(&self, x: &Tensor) -> Result<Tensor> {
        let x_dbl = self.x_proj.forward(x)?;
        let delta = self.dt_proj.forward(x)?;
        
        crate::kernels::selective_scan_custom(
            x,
            &delta,
            &self.a_log,
            &x_dbl, 
            &x_dbl, 
            &self.d
        )
    }
}

impl Module for ZumarMambaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. الإسقاط الأولي (768 -> 3072)
        let projected = self.in_proj.forward(x)?;
        
        // 2. تقسيم المصفوفة إلى مسارين (x_path و gate)
        // d_inner = 1536
        let chunks = projected.chunk(2, candle_core::D::Minus1)?;
        let x_path = &chunks[0];
        let gate = &chunks[1];

        // 3. تطبيق الـ SSM على المسار الأول
        let ssm_output = self.apply_ssm(x_path)?;
        
        // 4. دمج المسارين باستخدام تفعيل Sigmoid للـ Gate (SiLU approximation)
        // نستخدم sigmoid() كبديل آمن ومباشر
        let gate_activated = candle_nn::ops::sigmoid(gate)?;
        let x_gated = ssm_output.broadcast_mul(&gate_activated)?;

        // 5. الإسقاط النهائي للعودة إلى البعد الأصلي (768)
        self.out_proj.forward(&x_gated)
    }
}
