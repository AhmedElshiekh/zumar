// layers/config.rs - إدارة أبعاد النموذج الديناميكية

use serde::{Serialize, Deserialize};

/// تكوين النموذج الديناميكي
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub kv_heads: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

impl ModelConfig {
    /// إنشاء تكوين من عدد المعاملات المطلوب (بالمليارات)
    pub fn from_params_billions(params_b: f32) -> Self {
        let hidden_size = Self::estimate_hidden_size(params_b);
        let num_layers = Self::estimate_num_layers(hidden_size);
        let num_heads = hidden_size / 64; // head_dim = 64
        let kv_heads = (num_heads / 4).max(1);
        let num_experts = Self::estimate_num_experts(hidden_size);
        
        Self {
            vocab_size: 50257,
            hidden_size,
            num_layers,
            num_heads,
            kv_heads,
            num_experts,
            top_k: 2,
            max_seq_len: 4096,
            rope_theta: 10000.0,
        }
    }
    
    /// إنشاء تكوين من أبعاد مخصصة
    pub fn from_dimensions(hidden_size: usize, num_layers: usize, num_experts: usize) -> Self {
        let num_heads = (hidden_size / 64).max(1);
        Self {
            vocab_size: 50257,
            hidden_size,
            num_layers,
            num_heads,
            kv_heads: (num_heads / 4).max(1),
            num_experts,
            top_k: 2,
            max_seq_len: 4096,
            rope_theta: 10000.0,
        }
    }
    
    /// تقدير الحجم المخفي بناءً على عدد المعاملات
    fn estimate_hidden_size(params_b: f32) -> usize {
        match params_b {
            p if p < 0.1 => 128,      // 80M
            p if p < 0.5 => 512,      // 400M
            p if p < 1.0 => 768,      // 700M
            p if p < 3.0 => 1024,     // 1.5B
            p if p < 7.0 => 2048,     // 7B
            p if p < 13.0 => 4096,    // 13B
            p if p < 30.0 => 8192,    // 30B
            p if p < 70.0 => 12288,   // 70B
            _ => 16384,               // 100B-200B
        }
    }
    
    /// تقدير عدد الطبقات بناءً على الحجم المخفي
    fn estimate_num_layers(hidden_size: usize) -> usize {
        match hidden_size {
            s if s <= 128 => 6,
            s if s <= 512 => 12,
            s if s <= 1024 => 24,
            s if s <= 2048 => 32,
            s if s <= 4096 => 48,
            s if s <= 8192 => 64,
            s if s <= 12288 => 80,
            _ => 96,
        }
    }
    
    /// تقدير عدد الخبراء
    fn estimate_num_experts(hidden_size: usize) -> usize {
        match hidden_size {
            s if s <= 512 => 6,
            s if s <= 1024 => 8,
            s if s <= 2048 => 12,
            s if s <= 4096 => 16,
            s if s <= 8192 => 24,
            _ => 32,
        }
    }
    
    /// حساب عدد المعاملات التقريبي
    pub fn total_params(&self) -> usize {
        let embedding = self.vocab_size * self.hidden_size;
        let per_layer_attn = 4 * self.hidden_size * self.hidden_size;
        let per_layer_moe = self.num_experts * self.hidden_size * self.hidden_size;
        let lm_head = self.vocab_size * self.hidden_size;
        embedding + (self.num_layers * (per_layer_attn + per_layer_moe)) + lm_head
    }
    
    pub fn description(&self) -> String {
        format!(
            "{}d, {}L, {} heads, {} experts ({}M params)",
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.num_experts,
            self.total_params() / 1_000_000
        )
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::from_params_billions(0.083) // 83M
    }
}