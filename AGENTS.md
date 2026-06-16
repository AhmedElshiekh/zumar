# Zumar — Agent Instructions

## Project
Sovereign AI model with BitNet b1.58 (2-bit), Mamba SSM, Sparse MoE. Rust workspace with 3 crates: `core` (model/inference/training), `bridge` (HTTP API), `desktop` (placeholder).

## Key Commands
```sh
cargo run -p core               # chat (default subcommand)
cargo run -p core --release     # optimized chat
cargo run -p core --release --features blas  # CPU with OpenBLAS
cargo run -p core -- list-models
cargo run -p core -- list-teachers
cargo run -p core -- help

# Distillation pipeline (order: extract → distill)
cargo run -p core -- extract /data --teacher llama-13b --force
cargo run -p core -- distill 100 /data --teachers llama,jais --size 1.5B

# Export trained model
cargo run -p core -- pack
```

## Core Subcommands
Defined in `core/src/main.rs`. Default is `chat`. Others: `distill`, `train`, `extract`, `pack`, `list-models`, `list-teachers`, `help`. Custom dimensions via `--dims HIDDEN LAYERS EXPERTS`.

## Bridge
OpenAI-compatible API on `0.0.0.0:8080` (POST `/v1/chat/completions`). Run: `cargo run -p bridge`.

## Model & Data Layout
- `models/zumar/{size}B` or `{size}M/` — trained student models (model.safetensors, zumar-b1.58.zmr, zumar-b1.58.gguf)
- `models/teacher/` — teacher .safetensors files for distillation
- `models/zlog/` — extracted logits (.zlog format)
- `models/tokenizer/tokenizer.json` — GPT-2 based tokenizer (vocab=50257)
- `data/all_training.txt` — training data

## Architecture
- `layers/mod.rs:ZumarModel` — entrypoint, `layers/mod.rs:ZumarBlock` — per-layer (BitLinear + FlashAttention + MoE)
- `layers/bitlinear.rs` — `ZumarBitLinear` (1-bit ternary weights)
- `layers/mamba.rs` — SSM integration
- `layers/moe.rs` — Mixture of Experts with router
- `layers/lora.rs` — LoRA/QLoRA adapters
- `true_distill.rs` — multi-teacher distillation with EWC + VocabAlignment
- `loader.rs` — loads .zmr (fast 2-bit) or .safetensors (FP32)

## Weight Formats
- `.zmr` — custom 2-bit packed (50MB RAM for small models, direct inference)
- `.safetensors` — FP32 fallback (~970MB RAM)
- `.gguf` — llama.cpp compatible

## Features
`cuda`, `metal`, `accelerate`, `blas` (OpenBLAS). Default: CPU only.

## Notes
- Rust edition 2024, resolver "2"
- Comments and docs are in Arabic
- No tests, no CI, no formatter/linter config found
- `models/`, `data/`, `*.safetensors`, `*.gguf` are gitignored
