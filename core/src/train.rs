use candle_core::{Tensor, Result, Device, Var};
use candle_nn::{Optimizer, AdamW, ParamsAdamW};

pub struct ZumarTrainer {
    optimizer: AdamW,
}

impl ZumarTrainer {
    pub fn new(vars: Vec<Var>) -> Result<Self> {
        // إعداد المحسن (Optimizer) - AdamW هو الأفضل لتدريب الـ Transformers
        let params = ParamsAdamW::default();
        let optimizer = AdamW::new(vars, params)?;
        Ok(Self { optimizer })
    }

    pub fn train_step(&mut self, loss: &Tensor) -> Result<()> {
        // 1. حساب الاشتقاقات (Backpropagation)
        self.optimizer.backward_step(loss)?;
        Ok(())
    }
}
