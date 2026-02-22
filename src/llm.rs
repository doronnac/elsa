use std::num::NonZeroU32;
use std::pin::pin;

use anyhow::{Context, Result};
use log::{debug, info, trace};
use regex::Regex;
use serde::Deserialize;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

// ---------------------------------------------------------------------------
// LLM judge response
// ---------------------------------------------------------------------------

/// The structured JSON the LLM is expected to produce.
#[derive(Debug, Deserialize)]
pub struct LlmDecision {
    pub decision: String,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Chat message helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

impl std::fmt::Display for ChatMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]: {}", self.role, self.content)
    }
}

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

pub struct ModelConfig {
    /// How many layers to offload to GPU (0 = CPU only).
    pub n_gpu_layers: u32,
    /// Context window size in tokens.
    pub n_ctx: u32,
    /// Maximum tokens to generate per completion.
    pub max_tokens: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            n_ctx: 8092,
            max_tokens: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Sampler builders
// ---------------------------------------------------------------------------

fn build_free_sampler() -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::min_p(0.0, 1),
        LlamaSampler::temp(1.0),
        LlamaSampler::dist(1234),
    ])
}

fn build_sampler() -> Result<LlamaSampler> {
    Ok(LlamaSampler::chain_simple([
        LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::min_p(0.0, 1),
        LlamaSampler::temp(1.0),
        LlamaSampler::dist(1234),
    ]))
}

// ---------------------------------------------------------------------------
// LLM — loaded model handle
// ---------------------------------------------------------------------------

pub struct LLM {
    #[allow(dead_code)]
    backend: &'static LlamaBackend,
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    n_ctx: u32,
    max_tokens: usize,
}

impl LLM {
    pub fn load_model(model_path: &str, config: ModelConfig) -> Result<Self> {
        let backend: &'static LlamaBackend = Box::leak(Box::new(
            LlamaBackend::init().context("failed to init llama backend")?,
        ));

        info!("Loading model from: {model_path}");
        info!(
            "  config: n_gpu_layers={}, n_ctx={}, max_tokens={}",
            config.n_gpu_layers, config.n_ctx, config.max_tokens
        );

        let model_params = pin!(LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers));
        let model: &'static LlamaModel = Box::leak(Box::new(
            LlamaModel::load_from_file(backend, model_path, &model_params)
                .context("failed to load model")?,
        ));

        info!("Model loaded successfully");

        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(
            NonZeroU32::new(config.n_ctx).expect("n_ctx must be > 0"),
        ));
        let ctx = model
            .new_context(backend, ctx_params)
            .context("failed to create inference context")?;

        Ok(Self {
            backend,
            model,
            ctx,
            n_ctx: config.n_ctx,
            max_tokens: config.max_tokens,
        })
    }

    /// Run an unconstrained chat completion.
    #[allow(dead_code)]
    pub fn chat(&mut self, messages: &[ChatMessage]) -> Result<String> {
        let mut sampler = build_free_sampler();
        self.generate(messages, &mut sampler)
    }

    pub fn judge(
        &mut self,
        messages: &[ChatMessage],
        valid_choices: &[&str],
    ) -> Result<LlmDecision> {
        info!("Judging messages \n {messages:?}");

        // Initialize Sampler
        let mut sampler = build_sampler().context("Failed to initialize sampler")?;

        // Generate
        let raw = self.generate(messages, &mut sampler)?;

        // 4. Parse & Validate
        let decision = parse_decision(&raw)?;

        if valid_choices.contains(&decision.decision.as_str()) {
            info!(
                "Judge succeeded: {} (reason: {})",
                decision.decision, decision.reason
            );
            return Ok(decision);
        }

        anyhow::bail!(
            "Judge generated invalid decision '{}' (valid: {:?})",
            decision.decision,
            valid_choices
        );
    }
    /// Core generation: tokenize messages, feed prompt, sample tokens.
    fn generate(&mut self, messages: &[ChatMessage], sampler: &mut LlamaSampler) -> Result<String> {
        info!("=== LLM CALL: {} messages ===", messages.len());
        for (i, msg) in messages.iter().enumerate() {
            debug!("  msg[{i}] {msg}");
        }

        self.ctx.clear_kv_cache();

        let llama_msgs: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to create chat messages")?;

        let tmpl = self
            .model
            .chat_template(None)
            .context("model has no chat template")?;
        let prompt = self
            .model
            .apply_chat_template(&tmpl, &llama_msgs, true)
            .context("failed to apply chat template")?;

        trace!("=== RENDERED PROMPT ===\n{prompt}\n=== END PROMPT ===");

        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .context("tokenization failed")?;

        info!("Prompt tokenized: {} tokens", tokens.len());

        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        let last_idx = (tokens.len() - 1) as i32;
        for (i, tok) in (0i32..).zip(tokens.iter()) {
            batch.add(*tok, i, &[0], i == last_idx)?;
        }
        self.ctx
            .decode(&mut batch)
            .context("initial decode failed")?;

        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = batch.n_tokens();

        for _ in 0..self.max_tokens {
            let tok = sampler.sample(&self.ctx, batch.n_tokens() - 1);
            sampler.accept(tok);

            if self.model.is_eog_token(tok) {
                debug!("Hit EOG token, stopping generation");
                break;
            }

            let piece = self
                .model
                .token_to_piece(tok, &mut decoder, true, None)
                .context("token_to_piece failed")?;
            output.push_str(&piece);

            batch.clear();
            batch.add(tok, n_cur, &[0], true)?;
            self.ctx.decode(&mut batch).context("decode step failed")?;
            n_cur += 1;
        }

        info!(
            "=== LLM RAW OUTPUT ({} chars) ===\n{}\n=== END OUTPUT ===",
            output.len(),
            output
        );

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// JSON extraction
// ---------------------------------------------------------------------------

pub fn parse_decision(raw: &str) -> Result<LlmDecision> {
    let re_think = Regex::new(r"(?s)<think>(.*?)</think>").unwrap();
    for cap in re_think.captures_iter(raw) {
        let thought = cap.get(1).map_or("", |m| m.as_str()).trim();
        if !thought.is_empty() {
            debug!("Model thinking:\n{thought}");
        }
    }

    let cleaned = re_think.replace_all(raw, "");
    debug!("After stripping <think> blocks:\n{cleaned}");

    let re_json = Regex::new(r"(?s)\{[^{}]*\}").unwrap();
    let json_str = re_json.find(&cleaned).map(|m| m.as_str()).context(format!(
        "no JSON object found in LLM output. Raw output:\n{raw}"
    ))?;

    debug!("Extracted JSON: {json_str}");

    let decision: LlmDecision =
        serde_json::from_str(json_str).context(format!("failed to parse JSON: {json_str}"))?;

    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_decision_clean() {
        let raw = r#"{"decision": "FAILED_RUDE", "reason": "The user was hostile"}"#;
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.decision, "FAILED_RUDE");
    }

    #[test]
    fn test_parse_decision_with_think() {
        let raw = r#"<think>The user refused to show their passport and was rude.</think>
{"decision": "FAILED_RUDE", "reason": "Refused passport and was hostile"}"#;
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.decision, "FAILED_RUDE");
        assert!(d.reason.contains("hostile"));
    }

    #[test]
    fn test_parse_decision_with_surrounding_text() {
        let raw = r#"Here is my judgement:
<think>thinking hard...</think>
Based on the interaction, {"decision":"PASSPORT_CHECK","reason":"Cooperated nicely"}. That is my verdict."#;
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.decision, "PASSPORT_CHECK");
    }
}
