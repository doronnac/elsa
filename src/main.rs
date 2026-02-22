mod game;
mod llm;

use anyhow::{Context, Result};
use llm::{ModelConfig, LLM};

fn main() -> Result<()> {
    // Initialize logging. Control verbosity with RUST_LOG env var:
    //   RUST_LOG=info   cargo run -- model.gguf   # messages + transitions
    //   RUST_LOG=debug  cargo run -- model.gguf   # + judge instructions + parsed JSON
    //   RUST_LOG=trace  cargo run -- model.gguf   # + full rendered prompt template
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = std::env::args().collect();

    let model_path = args.get(1).context(
        "Usage: elsa <path-to-model.gguf> [gpu_layers] [context_size] [max_tokens]\n\
         \n\
         Example:\n  elsa ./models/qwen2.5-3b-instruct-q4_k_m.gguf 99 8092 1024\n\
         \n\
         Logging: set RUST_LOG=debug or RUST_LOG=trace for verbose output",
    )?;

    let config = ModelConfig {
        n_gpu_layers: args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0),
        n_ctx: args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8092),
        max_tokens: args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024),
    };

    println!("Loading model: {model_path}");
    println!("  GPU layers : {}", config.n_gpu_layers);
    println!("  Context    : {}", config.n_ctx);
    println!("  Max tokens : {}", config.max_tokens);

    let mut loaded_model = LLM::load_model(model_path, config).context("failed to load model")?;

    let tree = game::tree::airport_security_scenario();

    game::run(&mut loaded_model, tree)
}
