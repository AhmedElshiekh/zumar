use candle_core::{Tensor, Result};

pub struct KVCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    // تحديث الذاكرة بالكلمات الجديدة
    pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.k {
            None => k,
            Some(prev_k) => Tensor::cat(&[prev_k, &k], 1)?, // دمج المفاتيح الجديدة مع القديمة
        };
        let v = match &self.v {
            None => v,
            Some(prev_v) => Tensor::cat(&[prev_v, &v], 1)?, // دمج القيم الجديدة مع القديمة
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}
