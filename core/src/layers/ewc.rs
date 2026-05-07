use candle_core::{Tensor, Result, Device};
use candle_nn::{VarMap, Optimizer};
use std::collections::HashMap;
use std::path::Path;

// ════════════════════════════════════════════════════════════
// Fisher Information — أهمية كل وزن
// ════════════════════════════════════════════════════════════

/// معلومات Fisher لنقطة تدريب واحدة (بعد معلم واحد)
#[derive(Debug, Clone)]
pub struct FisherCheckpoint {
    /// Fisher information لكل وزن: اسم_المتغير → قيم Fisher
    pub fisher:      HashMap<String, Vec<f32>>,
    /// قيم الأوزان المثلى عند هذه النقطة θ*
    pub optimal_weights: HashMap<String, Vec<f32>>,
    /// اسم المعلم الذي أنتج هذه النقطة
    pub teacher_name: String,
    /// رقم الـ epoch عند الحفظ
    pub epoch:        usize,
}

/// Online EWC — يتراكم عبر جميع المعلمين
pub struct EWC {
    /// Fisher التراكمي: اسم_المتغير → مجموع Fisher من جميع المعلمين
    pub cumulative_fisher:  HashMap<String, Vec<f32>>,
    /// الأوزان المثلى المرجّحة من جميع المعلمين
    pub cumulative_optimal: HashMap<String, Vec<f32>>,
    /// تاريخ الـ checkpoints
    pub history:            Vec<FisherCheckpoint>,
    /// معامل EWC (كبير = لا ننسى، صغير = نتعلم بحرية)
    pub lambda:             f32,
    /// عدد المعلمين حتى الآن
    pub num_teachers:       usize,
}

impl EWC {
    pub fn new(lambda: f32) -> Self {
        Self {
            cumulative_fisher:  HashMap::new(),
            cumulative_optimal: HashMap::new(),
            history:            Vec::new(),
            lambda,
            num_teachers:       0,
        }
    }

    /// ✅ حساب Fisher Information بعد الانتهاء من معلم
    ///
    /// Fisher_i(θ) = E[ (∂log P(y|x,θ) / ∂θ)² ]
    /// تقريب عملي: متوسط مربع الـ gradients على عينة من البيانات
    pub fn compute_fisher(
        &self,
        varmap:   &VarMap,
        losses:   &[Tensor],   // عينة من الـ losses المحسوبة
        _device:   &Device,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let vars     = varmap.all_vars();
        let mut fisher: HashMap<String, Vec<f32>> = HashMap::new();

        // تهيئة Fisher بأصفار
        for var in &vars {
            let name  = format!("{:p}", var.as_tensor());
            let shape = var.as_tensor().dims().to_vec();
            let total: usize = shape.iter().product();
            fisher.insert(name, vec![0.0f32; total]);
        }

        // تراكم مربع الـ gradients
        let n = losses.len().max(1) as f32;
        for loss in losses {
            // backward pass
            let grads = loss.backward()?;

            for var in &vars {
                let name = format!("{:p}", var.as_tensor());
                if let Some(grad) = grads.get(var.as_tensor()) {
                    let grad_vals = grad.flatten_all()?.to_vec1::<f32>()?;
                    if let Some(f) = fisher.get_mut(&name) {
                        for (fi, gi) in f.iter_mut().zip(grad_vals.iter()) {
                            *fi += (gi * gi) / n;  // مربع الـ gradient
                        }
                    }
                }
            }
        }

        Ok(fisher)
    }

    /// ✅ تحديث Online EWC بعد الانتهاء من معلم
    pub fn update(
        &mut self,
        varmap:       &VarMap,
        losses:       &[Tensor],
        teacher_name: &str,
        epoch:        usize,
        device:       &Device,
    ) -> Result<()> {
        println!("\n   📐 Computing Fisher Information for '{}'...", teacher_name);

        let new_fisher  = self.compute_fisher(varmap, losses, device)?;
        let vars        = varmap.all_vars();

        // حفظ الأوزان المثلى الحالية θ*
        let mut optimal: HashMap<String, Vec<f32>> = HashMap::new();
        for var in &vars {
            let name = format!("{:p}", var.as_tensor());
            let vals = var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            optimal.insert(name.clone(), vals);
        }

        // ✅ Online EWC: تراكم Fisher
        // Fisher_total = Fisher_prev + Fisher_new
        for (name, new_f) in &new_fisher {
            let entry = self.cumulative_fisher
                .entry(name.clone())
                .or_insert_with(|| vec![0.0f32; new_f.len()]);
            for (acc, &nf) in entry.iter_mut().zip(new_f.iter()) {
                *acc += nf;
            }
        }

        // ✅ تحديث الأوزان المثلى المرجّحة
        // θ*_total = (n-1)/n * θ*_prev + 1/n * θ*_new
        self.num_teachers += 1;
        let n = self.num_teachers as f32;
        for (name, new_opt) in &optimal {
            let entry = self.cumulative_optimal
                .entry(name.clone())
                .or_insert_with(|| new_opt.clone());
            for (acc, &no) in entry.iter_mut().zip(new_opt.iter()) {
                *acc = *acc * (n - 1.0) / n + no / n;
            }
        }

        // حفظ في التاريخ
        self.history.push(FisherCheckpoint {
            fisher:          new_fisher,
            optimal_weights: optimal,
            teacher_name:    teacher_name.to_string(),
            epoch,
        });

        println!("   ✅ Fisher updated — {} teacher(s) consolidated", self.num_teachers);
        Ok(())
    }

    /// ✅ EWC Loss = λ * Σ_i F_i * (θ - θ*_i)²
    pub fn loss(&self, varmap: &VarMap, device: &Device) -> Result<Tensor> {
        if self.cumulative_fisher.is_empty() {
            return Tensor::new(&[0.0f32], device);
        }

        let vars     = varmap.all_vars();
        let mut total = 0.0f32;

        for var in &vars {
            let name  = format!("{:p}", var.as_tensor());
            let theta = var.as_tensor().flatten_all()?.to_vec1::<f32>()?;

            let fisher  = match self.cumulative_fisher.get(&name) {
                Some(f) => f,
                None    => continue,
            };
            let optimal = match self.cumulative_optimal.get(&name) {
                Some(o) => o,
                None    => continue,
            };

            // Σ F_i * (θ - θ*_i)²
            let penalty: f32 = fisher.iter()
                .zip(theta.iter())
                .zip(optimal.iter())
                .map(|((f, t), o)| f * (t - o).powi(2))
                .sum();

            total += penalty;
        }

        Tensor::new(&[total * self.lambda], device)
    }

    /// نفس EWC Loss لكن كـ Tensor يدعم backward
    pub fn loss_differentiable(&self, varmap: &VarMap, device: &Device) -> Result<Tensor> {
        if self.cumulative_fisher.is_empty() {
            return Tensor::new(&[0.0f32], device);
        }

        let vars = varmap.all_vars();
        let mut penalties: Vec<Tensor> = Vec::new();

        for var in &vars {
            let name  = format!("{:p}", var.as_tensor());
            let theta = var.as_tensor().flatten_all()?;

            let fisher  = match self.cumulative_fisher.get(&name) {
                Some(f) => f,
                None    => continue,
            };
            let optimal = match self.cumulative_optimal.get(&name) {
                Some(o) => o,
                None    => continue,
            };

            if theta.dim(0)? == 0 { continue; }

            let f_tensor = Tensor::from_vec(
                fisher.clone(), fisher.len(), device,
            )?;
            let o_tensor = Tensor::from_vec(
                optimal.clone(), optimal.len(), device,
            )?;

            // F * (θ - θ*)²
            let diff    = (&theta - &o_tensor)?;
            let penalty = (f_tensor * diff.powf(2.0)?)?
                .sum_all()?;

            penalties.push(penalty);
        }

        if penalties.is_empty() {
            return Tensor::new(&[0.0f32], device);
        }

        let total = penalties.iter()
            .skip(1)
            .fold(Ok(penalties[0].clone()), |acc, p| {
                acc.and_then(|a| &a + p)
            })?;

        total * self.lambda as f64
    }

    // ════════════════════════════════════════════════════════
    // Checkpoint: حفظ واستئناف
    // ════════════════════════════════════════════════════════

    /// حفظ حالة EWC كاملة
    // pub fn save(&self, path: &str) -> Result<()> {
    //     let state = EWCState {
    //         cumulative_fisher:  self.cumulative_fisher.clone(),
    //         cumulative_optimal: self.cumulative_optimal.clone(),
    //         lambda:             self.lambda,
    //         num_teachers:       self.num_teachers,
    //         history_names:      self.history.iter()
    //             .map(|h| h.teacher_name.clone())
    //             .collect(),
    //     };
    //     let json = serde_json::to_string(&state)
    //         .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    //     std::fs::create_dir_all(
    //         Path::new(path).parent().unwrap_or(Path::new("."))
    //     ).ok();
    //     std::fs::write(path, json)
    //         .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    //     println!("   💾 EWC state saved → {}", path);
    //     Ok(())
    // }

    pub fn save(&self, path: &str) -> Result<()> {
        let state = EWCState {
            cumulative_fisher:  self.cumulative_fisher.clone(),
            cumulative_optimal: self.cumulative_optimal.clone(),
            lambda:             self.lambda,
            num_teachers:       self.num_teachers,
            history_names:      self.history.iter().map(|h| h.teacher_name.clone()).collect(),
        };
        let encoded = rmp_serde::to_vec_named(&state)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        std::fs::create_dir_all(Path::new(path).parent().unwrap_or(Path::new("."))).ok();
        std::fs::write(path, &encoded)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        println!("   💾 EWC state saved → {}", path);
        Ok(())
    }

    /// تحميل حالة EWC
    // pub fn load(path: &str, lambda: f32) -> Result<Self> {
    //     if !Path::new(path).exists() {
    //         println!("   ℹ️  No EWC checkpoint found — starting fresh");
    //         return Ok(Self::new(lambda));
    //     }
    //     let json = std::fs::read_to_string(path)
    //         .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    //     let state: EWCState = serde_json::from_str(&json)
    //         .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    //     println!("   ♻️  EWC loaded — {} teacher(s) consolidated: {:?}",
    //         state.num_teachers, state.history_names);

    //     Ok(Self {
    //         cumulative_fisher:  state.cumulative_fisher,
    //         cumulative_optimal: state.cumulative_optimal,
    //         lambda:             state.lambda,
    //         num_teachers:       state.num_teachers,
    //         history:            Vec::new(),
    //     })
    // }
    
    pub fn load(path: &str, lambda: f32) -> Result<Self> {
        if !Path::new(path).exists() {
            println!("   ℹ️  No EWC checkpoint found — starting fresh");
            return Ok(Self::new(lambda));
        }
        let encoded = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let state: EWCState = rmp_serde::from_slice(&encoded)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
        println!("   ♻️  EWC loaded — {} teacher(s) consolidated: {:?}", state.num_teachers, state.history_names);
    
        Ok(Self {
            cumulative_fisher:  state.cumulative_fisher,
            cumulative_optimal: state.cumulative_optimal,
            lambda:             state.lambda,
            num_teachers:       state.num_teachers,
            history:            Vec::new(),
        })
    }

    /// معدل التعلم المناسب بحسب عدد المعلمين السابقين
    /// كلما تعلمنا من معلمين أكثر، قللنا الـ LR
    pub fn recommended_lr(&self, base_lr: f64) -> f64 {
        let decay = 0.7_f64.powi(self.num_teachers as i32);
        (base_lr * decay).max(1e-6)
    }

    /// تقرير الحالة
    pub fn report(&self) {
        println!("\n   📐 EWC Status:");
        println!("     Teachers consolidated: {}", self.num_teachers);
        println!("     Lambda:                {}", self.lambda);
        println!("     Recommended LR decay:  {:.4}", 0.7_f64.powi(self.num_teachers as i32));
        for h in &self.history {
            println!("     ✅ {} (epoch {})", h.teacher_name, h.epoch);
        }
    }
}

// ════════════════════════════════════════════════════════════
// Checkpoint كامل للمشروع
// ════════════════════════════════════════════════════════════

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DistillCheckpoint {
    /// رقم المعلم الذي وصلنا إليه
    pub teacher_index: usize,
    /// اسم المعلم الحالي
    pub teacher_name:  String,
    /// رقم الـ epoch داخل المعلم الحالي
    pub epoch:         usize,
    /// إجمالي الـ epochs المنجزة
    pub total_epochs:  usize,
    /// أفضل loss حتى الآن
    pub best_loss:     f32,
    /// معدل التعلم الحالي
    pub current_lr:    f64,
    /// قائمة المعلمين المنجزين
    pub done_teachers: Vec<String>,
}

impl DistillCheckpoint {
    pub fn new() -> Self {
        Self {
            teacher_index: 0,
            teacher_name:  String::new(),
            epoch:         0,
            total_epochs:  0,
            best_loss:     f32::INFINITY,
            current_lr:    0.001,
            done_teachers: Vec::new(),
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        std::fs::create_dir_all(
            Path::new(path).parent().unwrap_or(Path::new("."))
        ).ok();
        std::fs::write(path, json)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        if !Path::new(path).exists() {
            return Ok(Self::new());
        }
        let json = std::fs::read_to_string(path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        serde_json::from_str(&json)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    pub fn mark_teacher_done(&mut self, name: &str) {
        self.done_teachers.push(name.to_string());
        self.teacher_index += 1;
        self.epoch          = 0;
    }

    pub fn is_teacher_done(&self, name: &str) -> bool {
        self.done_teachers.contains(&name.to_string())
    }
}

// ════════════════════════════════════════════════════════════
// Serde structs داخلية
// ════════════════════════════════════════════════════════════

#[derive(serde::Serialize, serde::Deserialize)]
struct EWCState {
    cumulative_fisher:  HashMap<String, Vec<f32>>,
    cumulative_optimal: HashMap<String, Vec<f32>>,
    lambda:             f32,
    num_teachers:       usize,
    history_names:      Vec<String>,
}
