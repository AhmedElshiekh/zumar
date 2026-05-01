use candle_core::{Tensor, Result};

/// KV Cache مع حد أقصى للطول (Sliding Window)
/// بدون حد: الذاكرة تتضخم إلى ما لا نهاية في المحادثات الطويلة
pub struct KVCache {
    pub k:           Option<Tensor>,
    pub v:           Option<Tensor>,
    pub max_seq_len: usize,   // الحد الأقصى للتسلسل المحفوظ
    pub current_len: usize,   // الطول الحالي المخزّن
}

impl KVCache {
    /// `max_seq_len`: الحد الأقصى قبل بدء الـ sliding window
    /// القيمة الافتراضية المقترحة: 2048
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            max_seq_len,
            current_len: 0,
        }
    }

    pub fn with_default_size() -> Self {
        Self::new(2048)
    }

    /// تحديث الكاش بالـ K, V الجديدة
    /// إذا تجاوز الطول `max_seq_len` يتم قطع أقدم الـ tokens (sliding window)
    pub fn update(&mut self, k_new: Tensor, v_new: Tensor) -> Result<(Tensor, Tensor)> {
        // ── دمج مع الكاش السابق ──────────────────────────────────
        let k_full = match &self.k {
            None       => k_new,
            Some(prev) => Tensor::cat(&[prev, &k_new], 1)?,
        };
        let v_full = match &self.v {
            None       => v_new,
            Some(prev) => Tensor::cat(&[prev, &v_new], 1)?,
        };

        // ── Sliding Window: قطع من البداية إذا تجاوز الحد ──────
        let seq_len = k_full.dim(1)?;

        let (k_stored, v_stored) = if seq_len > self.max_seq_len {
            // احتفظ بآخر max_seq_len token فقط
            let start = seq_len - self.max_seq_len;
            (
                k_full.narrow(1, start, self.max_seq_len)?,
                v_full.narrow(1, start, self.max_seq_len)?,
            )
        } else {
            (k_full, v_full)
        };

        self.current_len = k_stored.dim(1)?;
        self.k           = Some(k_stored.clone());
        self.v           = Some(v_stored.clone());

        Ok((k_stored, v_stored))
    }

    /// عدد الـ tokens المخزنة حالياً
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// هل الكاش فارغ؟
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// هل وصلنا للحد الأقصى؟
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_seq_len
    }

    /// نسبة الامتلاء (للـ logging)
    pub fn usage_ratio(&self) -> f32 {
        self.current_len as f32 / self.max_seq_len as f32
    }

    pub fn reset(&mut self) {
        self.k           = None;
        self.v           = None;
        self.current_len = 0;
    }
}

/// كاش لجميع الطبقات
pub struct LayerKVCache {
    pub caches:      Vec<KVCache>,
    pub max_seq_len: usize,
}

impl LayerKVCache {
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            caches: (0..num_layers).map(|_| KVCache::new(max_seq_len)).collect(),
            max_seq_len,
        }
    }

    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut KVCache> {
        self.caches.get_mut(layer_idx)
    }

    pub fn reset_all(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// أقصى طول مستخدم عبر جميع الطبقات
    pub fn max_used_len(&self) -> usize {
        self.caches.iter().map(|c| c.current_len).max().unwrap_or(0)
    }

    pub fn usage_report(&self) -> String {
        let used = self.max_used_len();
        format!("KV Cache: {}/{} ({:.0}%)", used, self.max_seq_len,
            used as f32 / self.max_seq_len as f32 * 100.0)
    }
}
