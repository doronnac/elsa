use std::io::{self, Write};

use anyhow::Result;
use log::{debug, info, warn};

use crate::game_tree::{GameNode, GameTree};
use crate::llm::{ChatMessage, LLM};

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------

struct GameState {
    tree: GameTree,
    current_node_id: String,
    /// Conversation turns only (assistant + user messages). System messages
    /// are injected at query time to avoid confusing small models with stale
    /// instructions from previous steps.
    conversation: Vec<ChatMessage>,
    /// Number of non-terminal steps the player has completed.
    steps_completed: usize,
}

impl GameState {
    fn new(tree: GameTree) -> Self {
        let start_id = tree.start_node_id.clone();
        Self {
            tree,
            current_node_id: start_id,
            conversation: Vec::new(),
            steps_completed: 0,
        }
    }

    fn current_node(&self) -> &GameNode {
        self.tree.get(&self.current_node_id).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/// Short, direct system prompt. Small models do better with brief instructions
/// that don't contain meta-commentary about rules or JSON schemas.
const SYSTEM_PROMPT: &str = "\
You are a border security guard at an airport. You are having a conversation with a traveller. Your job is to categorize the Traveller's last response based on the following rules:";

/// Build the complete message list for an LLM judge call.
///
/// Structure (kept minimal for small models):
///   [system] brief role prompt
///   [assistant] guard line 1
///   [user] traveller response 1
///   ...ongoing conversation...
///   [system] judge instruction with decision criteria
///
/// Previous-node system_context is NOT included — only the current
/// node's criteria appear, right before the model generates.
fn build_judge_messages(state: &GameState, node: &GameNode) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // 1. General system prompt + Judge instructions
    messages.push(ChatMessage::system(format!(
        "{SYSTEM_PROMPT} \n {}",
        build_judge_instruction(node)
    )));

    // 2. Conversation so far (assistant + user turns only)
    messages.extend(state.conversation.clone());

    messages
}

/// Build the judge instruction. Kept short and direct:
/// - States what the guard just asked
/// - Lists the PASS option first, then the FAIL option
/// - Asks for a one-line JSON
fn build_judge_instruction(node: &GameNode) -> String {
    let mut s = String::new();

    if let Some(ctx) = &node.system_context {
        s.push_str(ctx);
        s.push_str("\n\n");
    }

    let options: Vec<&str> = node.next_node_ids.iter().map(|s| s.as_str()).collect();
    let options_str = options.join(", ");

    s.push_str(&format!(
        "Pick one: {options_str}\n\
         Reply with JSON only: {{\"decision\": \"<PICK>\", \"reason\": \"<why>\"}}. JSON must be valid."
    ));

    s
}

// ---------------------------------------------------------------------------
// Game over screen
// ---------------------------------------------------------------------------

/// Outcome of a single game round.
enum GameOutcome {
    /// Player reached a terminal node.
    Finished {
        success: bool,
        steps_completed: usize,
        total_steps: usize,
        terminal_node_id: String,
    },
    /// Player typed quit mid-game.
    Quit,
}

fn show_game_over(outcome: &GameOutcome) {
    println!("\n========================================");
    println!("             GAME OVER");
    println!("========================================");

    match outcome {
        GameOutcome::Finished {
            success,
            steps_completed,
            total_steps,
            terminal_node_id,
        } => {
            if *success {
                println!("  Result: CLEARED - You passed border control!");
            } else {
                println!("  Result: DENIED - You were stopped at the border.");
            }
            println!(
                "  Score:  {} / {} steps completed",
                steps_completed, total_steps
            );
            println!("  Ended at: {}", terminal_node_id);
        }
        GameOutcome::Quit => {
            println!("  You walked away from the border control booth.");
        }
    }

    println!("========================================\n");
    println!("  [r] Restart    [q] Quit\n");
}

/// Read the player's post-game choice. Returns `true` to restart, `false` to quit.
fn prompt_restart() -> Result<bool> {
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        match input.trim().to_lowercase().as_str() {
            "r" => return Ok(true),
            "q" => return Ok(false),
            _ => println!("  Press [r] to restart or [q] to quit."),
        }
    }
}

// ---------------------------------------------------------------------------
// Single game round
// ---------------------------------------------------------------------------

fn play_round(model: &mut LLM, tree: &GameTree) -> Result<GameOutcome> {
    let mut state = GameState::new(tree.clone());
    let total_steps = tree.total_steps();

    info!("Game started. Initial node: {}", state.current_node_id);

    loop {
        let node = state.current_node().clone();
        info!(
            "Current node: {} (terminal={}, next={:?})",
            node.id, node.terminal, node.next_node_ids
        );

        // Display the guard's line
        println!("\n[Guard]: {}", node.transcript);

        // Add the guard's transcript to conversation
        state
            .conversation
            .push(ChatMessage::assistant(&node.transcript));

        // Terminal node -> game over
        if node.terminal {
            info!(
                "Game over at node: {} (success={})",
                node.id, node.is_success
            );
            return Ok(GameOutcome::Finished {
                success: node.is_success,
                steps_completed: state.steps_completed,
                total_steps,
                terminal_node_id: node.id.clone(),
            });
        }

        // Read user input
        print!("\n[You]: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.is_empty() {
            println!("(Please say something.)");
            state.conversation.pop();
            continue;
        }

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            return Ok(GameOutcome::Quit);
        }

        info!("User input: \"{input}\"");

        // Add user response to conversation
        state.conversation.push(ChatMessage::user(&input));

        // Build messages and judge
        let messages = build_judge_messages(&state, &node);
        debug!(
            "Judge messages ({} total):\n{}",
            messages.len(),
            messages
                .iter()
                .enumerate()
                .map(|(i, m)| format!("  msg[{i}] {m}"))
                .collect::<Vec<_>>()
                .join("\n")
        );

        // Valid choices for the grammar-constrained judge
        let valid_choices: Vec<&str> = node.next_node_ids.iter().map(|s| s.as_str()).collect();

        println!("\n(Thinking...)");
        let decision = model.judge(&messages, &valid_choices)?;

        // Grammar ensures decision is valid, but keep a safety check
        if !node.next_node_ids.contains(&decision.decision) {
            warn!(
                "LLM chose '{}' which is not in {:?}. Falling back to first option.",
                decision.decision, node.next_node_ids
            );
            let fallback = node.next_node_ids.first().unwrap().clone();
            state.current_node_id = fallback.clone();
            info!("Fallback transition: {} -> {}", node.id, fallback);
        } else {
            info!(
                "Transition: {} -> {} (reason: {})",
                node.id, decision.decision, decision.reason
            );
            state.current_node_id = decision.decision.clone();
        }

        // Player survived this step
        state.steps_completed += 1;

        // Show the LLM's reasoning
        println!("(Judge reasoning: {})", decision.reason);
    }
}

// ---------------------------------------------------------------------------
// Public entry point — runs games in a loop until the player quits
// ---------------------------------------------------------------------------

pub fn run(model: &mut LLM, tree: GameTree) -> Result<()> {
    loop {
        println!("\n========================================");
        println!("   AIRPORT BORDER CONTROL SIMULATOR");
        println!("========================================");
        println!("Try to pass through border control.");
        println!("Type your responses naturally.\n");

        let outcome = play_round(model, &tree)?;
        show_game_over(&outcome);

        if !prompt_restart()? {
            println!("Thanks for playing!");
            break;
        }

        info!("Player chose to restart");
    }

    Ok(())
}
