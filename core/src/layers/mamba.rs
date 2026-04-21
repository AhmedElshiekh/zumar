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

        // إسقاط ينتج 4096 (2048 للمسار و 2048 للبوابة)
        let in_proj = candle_nn::linear(cfg.d_model, d_inner * 2, vs.pp("in_proj"))?;
        let x_proj = candle_nn::linear(d_inner, cfg.d_state * 2 + d_inner, vs.pp("x_proj"))?;
        let dt_proj = candle_nn::linear(d_inner, d_inner, vs.pp("dt_proj"))?;
        let out_proj = candle_nn::linear(d_inner, cfg.d_model, vs.pp("out_proj"))?;

        let a_log = vs.get((16, d_inner), "a_log")?;
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
        // 1. الإسقاط لـ 4096
        let projected = self.in_proj.forward(x)?;
        
        // 2. التقسيم لنصفين متساويين (2048 لكل منهما)
        let chunks = projected.chunk(2, candle_core::D::Minus1)?;
        let x_path = &chunks[0];
        let gate_path = &chunks[1];

        // 3. معالجة المسار الأول (SSM)
        let ssm_output = self.apply_ssm(x_path)?;

        // 4. تفعيل البوابة باستخدام SiLU ودمج المسارين
        let gate_activated = candle_nn::ops::silu(gate_path)?;
        let combined = ssm_output.broadcast_mul(&gate_activated)?;

        // 5. الإسقاط النهائي للعودة لـ 1024
        self.out_proj.forward(&combined)
    }
}
