use candle_core::{Tensor, Result};
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
    pub d_inner: usize,
}

impl ZumarMambaBlock {
    pub fn new(cfg: &ZumarMambaConfig, vs: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_model * cfg.expand;
        let device = vs.device();

        let in_proj = candle_nn::linear(cfg.d_model, d_inner * 2, vs.pp("in_proj"))?;
        let x_proj = candle_nn::linear(d_inner, cfg.d_state * 2 + d_inner, vs.pp("x_proj"))?;
        let dt_proj = candle_nn::linear(d_inner, d_inner, vs.pp("dt_proj"))?;
        let out_proj = candle_nn::linear(d_inner, cfg.d_model, vs.pp("out_proj"))?;

        let a_log = vs.get((cfg.d_state, d_inner), "a_log")?;
        let d = vs.get(d_inner, "d")?;
        let conv1d = Tensor::zeros((d_inner, cfg.d_conv), candle_core::DType::F32, device)?;

        Ok(Self {
            in_proj,
            _conv1d: conv1d,
            x_proj,
            dt_proj,
            out_proj,
            a_log,
            d,
            _d_state: cfg.d_state,
            d_inner,
        })
    }

    fn softplus(x: &Tensor) -> Result<Tensor> {
        let exp_x = x.exp()?;
        let one = Tensor::ones_like(&exp_x)?;
        (one + exp_x)?.log()
    }

    /// SSM مبسط يعمل
    pub fn apply_ssm(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let device = x.device();

        let x_dbl = self.x_proj.forward(x)?;
        let delta = self.dt_proj.forward(x)?;
        let delta = Self::softplus(&delta)?;

        let b_split = x_dbl.narrow(2, 0, self._d_state)?;
        let c_split = x_dbl.narrow(2, self._d_state, self._d_state)?;
        let x_rem = x_dbl.narrow(2, self._d_state * 2, d)?;

        let mut h = Tensor::zeros((b, self._d_state, d), candle_core::DType::F32, device)?;
        let mut outputs = Vec::new();

        for t in 0..s {
            let b_t = b_split.get_on_dim(1, t)?.unsqueeze(2)?;
            let x_t = x_rem.get_on_dim(1, t)?.unsqueeze(1)?;

            h = (h * 0.95f64)?;
            h = (h + b_t.broadcast_mul(&x_t)?)?;

            let c_t = c_split.get_on_dim(1, t)?.unsqueeze(1)?;
            let ch = c_t.matmul(&h)?;
            let d_val = self.d.unsqueeze(0)?.unsqueeze(0)?;
            let dx = x_t.broadcast_mul(&d_val)?;
            let y = (ch + dx)?;

            outputs.push(y);
        }

        let output = Tensor::cat(
            &outputs.iter().map(|t| t as &Tensor).collect::<Vec<_>>(),
            1,
        )?;

        Ok(output)
    }
}

impl Module for ZumarMambaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_rank = x.rank();
        let x_3d = if original_rank == 2 { x.unsqueeze(0)? } else { x.clone() };

        let projected = self.in_proj.forward(&x_3d)?;
        let chunks = projected.chunk(2, candle_core::D::Minus1)?;
        let x_path = &chunks[0];
        let gate_path = &chunks[1];

        let ssm_output = self.apply_ssm(x_path)?;
        let gate = candle_nn::ops::silu(gate_path)?;
        let combined = ssm_output.broadcast_mul(&gate)?;

        let output = self.out_proj.forward(&combined)?;

        if original_rank == 2 {
            output.squeeze(0)
        } else {
            Ok(output)
        }
    }
}