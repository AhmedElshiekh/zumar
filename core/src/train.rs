use candle_core::{Result, Device};
use candle_nn::VarMap;
use crate::layers::ZumarModel;
use crate::data::TrainingData;
use crate::distill::{Distiller, DistillConfig};

pub fn run_training(
    model: &mut ZumarModel,
    varmap: &VarMap,
    device: &Device,
    data_path: Option<&str>,
    epochs: usize,
) -> Result<()> {
    println!("============================================================");
    println!("🧬 ZUMAR TRAINING (Python-compatible style)");
    println!("============================================================");
    
    let training_data = TrainingData::load(data_path);
    let all_texts = training_data.repeat(10);
    
    println!("📊 {} texts", all_texts.len());
    
    let config = DistillConfig { epochs, learning_rate: 0.001 };
    let distiller = Distiller::new(config, device.clone());
    distiller.distill(model, varmap, &all_texts)?;
    
    Ok(())
}