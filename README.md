# ❄️ Elsa

**Elsa** is an experimental, text-based Role-Playing Game (RPG) where your survival depends on social intuition and situational awareness. 

This project is a **Proof of Concept (PoC)** for a localized language-learning platform. It uses a Small Language Model (SLM) running locally to act as a "Logic Judge," determining if your behavior is appropriate for a given scenario.

---

## 🚀 Getting Started

### Prerequisites
- Rust installed (Binaries coming soon).
- A GGUF model file (recommended: [SmolLM3-Q4_K_M.gguf](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf?download=true)).

### Installation
   ```bash
   # Clone
   git clone [https://github.com/doronnac/elsa.git](https://github.com/doronnac/elsa.git)
   cd elsa
   
   # Run
   cargo run --release [path-to-model-file]
   ```
---

### 🎮 The Game Loop

So far only a single, brief scenario has been built. This loop represents the broader vision for the engine.

1. **The Scenario:** You are placed in a high-stakes environment (e.g., Airport Security, a Job Interview, or a Medical Check-in).
2. **The Interaction:** You respond to an NPC (Non-Player Character).
3. **The Judgment:** A local SLM analyzes your response for logic, tone, and consistency.
4. **The Branch:** Based on the Judge's decision, the game either proceeds or moves into a "Suspicious" or "Failed" state.

---

### 🧠 Technical Architecture

Elsa is built to be private, fast, and entirely local. By using **llama.cpp**, the game runs without an internet connection or API fees. I specifically tested language models with the goal of going as small as possible, and ended up choosing SmolLM3 for several reasons:
- Reasoning models performed better for this task than much larger Instruct counterparts.
- Out of the reasoning models - SmolLM3 seems to have been the best compromise between accuracy, speed, size and language support.

---

### 🧩 Technical Challenges

The biggest hurdle so far is *consistency*, though current results are reassuringly positive. Even without exhausting advanced "correctness" mechanisms, the model responds surprisingly well to straightforward prompt engineering.

However, as scenarios increase in size and divergence, I fully expect the "logic drift" to exacerbate. To combat this, the project will place a heavy emphasis on adopting systems to ensure 100% reliability:

- GBNF Sampling: Using grammar-based sampling to force the LLM to output valid JSON every time.

- Prompt Iteration: Moving from simple instructions to structured Chain-of-Thought (CoT) prompting.

- Retry Logic & Consensus: Implementing multiple "judging" passes for critical game-state transitions.

- Strict SerDe: Robust serialization/deserialization to handle model outputs programmatically.

- Fine-Tuning: Exploring LoRA or full fine-tunes on smaller models (1B-3B) specifically for logic validation tasks.

---

### 🤔 Ok but why?

We all want to know many languages, but actually learning them can be really boring.

I’ve tried many apps, and they are usually one of three things: too slow, mind-numbingly repetitive, or "AI-enabled" slop designed to trick you into a subscription. My goal with Elsa is to see how far I can push the boundaries of fully-local LLMs.

So far, the results show that you don't need a massive cloud API to create a meaningful, high-stakes learning environment.
