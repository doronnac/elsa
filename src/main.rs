mod game;
mod llm;

use anyhow::{Context, Result};
use llm::{ModelConfig, LLM};

use crate::game::tree::GameTree;

fn main() -> Result<()> {
    // Initialize logging. Control verbosity with RUST_LOG env var:
    //   RUST_LOG=info   cargo run -- model.gguf   # messages + transitions
    //   RUST_LOG=debug  cargo run -- model.gguf   # + judge instructions + parsed JSON
    //   RUST_LOG=trace  cargo run -- model.gguf   # + full rendered prompt template
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stdout)
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = std::env::args().collect();

    let model_path = args.get(1).context(
        "
        Usage: cargo run <path-to-model.gguf> <path-to-scenario.json>
        \n\
        Example:\n  cargo run ./SmolLM3-Q4_K_M ./scenarios/airport.json \n\
    ",
    )?;

    let scenario = args.get(2).context(
        "
        Usage: cargo run <path-to-model.gguf> <path-to-scenario.json>
        \n\
        Example:\n  cargo run ./SmolLM3-Q4_K_M ./scenarios/airport.json \n\
    ",
    )?;

    let config = ModelConfig {
        n_gpu_layers: 0,
        n_ctx: 8092,
        max_tokens: 1024,
    };

    eprintln!("Loading model: {model_path}");
    eprintln!("GPU layers : {}", config.n_gpu_layers);
    eprintln!("Context    : {}", config.n_ctx);
    eprintln!("Max tokens : {}", config.max_tokens);

    let mut model = LLM::load_model(model_path, config).context("failed to load model")?;

    let game_tree: GameTree = serde_json::from_str(&std::fs::read_to_string(scenario)?)?;

    game::run(&mut model, game_tree)
}
