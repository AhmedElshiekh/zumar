use candle_core::{Tensor, Result, Device, Module};
use candle_nn::VarMap;
use crate::layers::ZumarModel;
use std::time::Instant;

pub struct DistillConfig {
    pub epochs: usize,
    pub learning_rate: f64,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self { epochs: 20, learning_rate: 0.01 }
    }
}

pub struct Distiller {
    config: DistillConfig,
    device: Device,
}

impl Distiller {
    pub fn new(config: DistillConfig, device: Device) -> Self {
        Self { config, device }
    }
    
    pub fn distill(
        &self,
        model: &mut ZumarModel,
        varmap: &VarMap,
        data: &[String],
    ) -> Result<()> {
        println!("\n🎓 Training (SGD - low memory)...");
        println!("   Epochs: {}", self.config.epochs);
        println!("   Samples: {}", data.len());
        
        let lr = self.config.learning_rate;
        let vars = varmap.all_vars();
        
        let start = Instant::now();
        
        for epoch in 0..self.config.epochs {
            let mut loss_sum = 0.0f32;
            let mut count = 0u32;
            
            for text in data.iter() {
                let tokens: Vec<u32> = text.chars()
                    .map(|c| (c as u32).wrapping_add(3))
                    .collect();
                
                if tokens.len() < 2 { continue; }
                
                for i in 0..tokens.len() - 1 {
                    let token_id = tokens[i];
                    let target_id = tokens[i + 1];
                    
                    // Forward
                    let input = Tensor::new(&[token_id], &self.device)?;
                    let emb = model.embedding.forward(&input)?.unsqueeze(0)?;
                    
                    let mut h = emb;  // emb moves into h
                    for layer in &mut model.layers {  // أضف &mut
                        h = layer.forward(&h)?;
                    }
                    h = model.final_norm.forward(&h)?;
                    let logits = model.lm_head.forward(&h)?.flatten_all()?;
                    
                    // Loss
                    let target = Tensor::new(&[target_id as i64], &self.device)?;
                    let loss = candle_nn::loss::cross_entropy(
                        &logits.unsqueeze(0)?, &target
                    )?;
                    
                    let loss_val = loss.to_scalar::<f32>()?;
                    loss_sum += loss_val;
                    count += 1;
                    
                    // Backward
                    let grads = loss.backward()?;
                    
                    // SGD update
                    for var in vars.iter() {
                        if let Some(grad) = grads.get(var) {
                            let step = (grad * lr)?;
                            let updated = var.sub(&step)?;
                            var.set(&updated)?;
                        }
                    }
                    
                    // Free memory (emb already moved to h)
                    drop(input);
                    drop(h);
                    drop(logits);
                    drop(target);
                    drop(loss);
                    drop(grads);
                    
                    if count % 100 == 0 {
                        print!("\r  Ep {} | Step {} | Loss {:.4}   ", 
                            epoch + 1, count, loss_sum / count as f32);
                    }
                }
            }
            
            let avg = loss_sum / count.max(1) as f32;
            println!();
            println!("  ✅ Ep {}: Loss {:.4} | {:.1}s", 
                epoch + 1, avg, start.elapsed().as_secs_f64());
        }
        
        println!("\n⏱ Total: {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}