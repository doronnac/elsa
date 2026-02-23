# ❄️ ELSA - Embedded Language Study Agent

**ELSA** is an experimental engine that can "play out" a schema based decision tree according to free-form input.

---

### Motivation


Research regarding Scenario Based Learning (SBL) shows increased efficacy when learning under simulated scenarios.

---

## 🚀 Getting Started

### Prerequisites
- Rust.
- A GGUF model file (recommended: [SmolLM3-Q4_K_M.gguf](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf?download=true)).

### Installation
   ```bash
   git clone https://github.com/doronnac/elsa.git && cd elsa
   
   RUST_LOG=info cargo run --release [path-to-model-file] 2>/dev/null
   ```
   
---

### Game Loop

1. **The Scenario:** You are placed in a high-stakes environment (e.g., Airport Security, a Job Interview, or a Medical Check-in).
2. **The Interaction:** You respond to an NPC (Non-Player Character).
3. **The Judgment:** A local SLM analyzes your response for logic, tone, and consistency.
4. **The Branch:** Based on the Judge's decision, the game either proceeds or moves into a "Suspicious" or "Failed" state.

---

### Technical Architecture

ELSA is built to be portable, fast and local.

I specifically tested language models with the goal of going as small as possible, and ended up choosing SmolLM3 for several reasons:
- Reasoning models performed better for this task than much larger Instruct counterparts.
- Out of the reasoning models - SmolLM3 seems to have been the best compromise between accuracy, speed, size and language support.

---

### Technical Challenges

The biggest hurdle so far is *consistency*, though current results are reassuringly positive; Even without exhausting advanced "correctness" mechanisms, the model responds surprisingly well to straightforward prompt engineering.

However, as scenarios increase in size and divergence, I fully expect the "logic drift" to exacerbate. To combat this, the project will place a heavy emphasis on adopting systems to ensure 100% reliability:

- GBNF Sampling: Using grammar-based sampling to force the LLM to output valid JSON every time.

- Prompt Iteration: Moving from simple instructions to structured Chain-of-Thought (CoT) prompting.

- Retry Logic & Consensus: Implementing multiple "judging" passes for critical game-state transitions.

- Strict SerDe: Robust serialization/deserialization to handle model outputs programmatically.

- Fine-Tuning: Exploring LoRA or full fine-tunes on smaller models (1B-3B) specifically for logic validation tasks.

---
