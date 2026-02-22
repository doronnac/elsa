use serde::Deserialize;

/// A single node in the game's decision tree.
#[derive(Debug, Clone, Deserialize)]
pub struct GameNode {
    /// Unique identifier for this node (e.g. "START", "QUESTION_1", "FAILED").
    pub id: String,
    /// The line the guard says when entering this node.
    pub transcript: String,
    // Terminal, success, next_node, etc
    pub node_type: NodeType,
    /// Extra system-prompt context injected when the game reaches this node.
    /// Gives the LLM roleplay instructions specific to this stage.
    pub system_context: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NextNode {
    // ID of an existing node
    pub id: String,
    // Useful for the LLM's system prompt
    pub description: String,
}

#[derive(Debug, Clone, Deserialize)]
pub enum NodeType {
    // Terminal node (is_success)
    Terminal(bool),
    // Decision node
    Decision(Vec<NextNode>),
}
