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
    pub _conv1d: Tensor,   // أضفنا _ لإخفاء التحذير
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub out_proj: Linear,
    pub a_log: Tensor,
    pub d: Tensor,
    pub _d_state: usize,   // أضفنا _ لإخفاء التحذير
}

impl ZumarMambaBlock {
    pub fn new(cfg: &ZumarMambaConfig, vs: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_model * cfg.expand;
        let device = vs.device();

        let in_proj = candle_nn::linear(cfg.d_model, d_inner * 2, vs.pp("in_proj"))?;
        let x_proj = candle_nn::linear(d_inner, d_inner + cfg.d_state * 2, vs.pp("x_proj"))?;
        let dt_proj = candle_nn::linear(d_inner, d_inner, vs.pp("dt_proj"))?;
        let out_proj = candle_nn::linear(d_inner, cfg.d_model, vs.pp("out_proj"))?;

        let a_log = vs.get((cfg.d_state, d_inner), "a_log")?;
        let d = vs.get(d_inner, "d")?;
        
        // استخدام d_inner و cfg.d_conv لإنشاء التنسور
        let conv1d = Tensor::zeros((d_inner, cfg.d_conv), DType::F32, device)?;

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
        let projected = self.in_proj.forward(x)?;
        let ssm_output = self.apply_ssm(&projected)?;
        self.out_proj.forward(&ssm_output)
    }
}
