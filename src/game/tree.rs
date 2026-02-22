use std::collections::HashMap;

use serde::Deserialize;

use crate::game::node::{GameNode, NextNode, NodeType};

/// The full scenario tree: a map of node-id -> GameNode.
#[derive(Debug, Clone, Deserialize)]
pub struct GameTree {
    pub nodes: HashMap<String, GameNode>,
    pub start_node_id: String,
}

impl GameTree {
    pub fn get(&self, id: &str) -> Option<&GameNode> {
        self.nodes.get(id)
    }

    /// Count the number of non-terminal nodes on the longest path through
    /// the tree (i.e. the maximum possible steps a player can complete).
    pub fn total_steps(&self) -> usize {
        self.longest_path(&self.start_node_id)
    }

    fn longest_path(&self, node_id: &str) -> usize {
        let node = match self.nodes.get(node_id) {
            Some(n) => n,
            None => return 0,
        };
        match node.node_type.clone() {
            NodeType::Terminal(_) => {
                return 0;
            }
            NodeType::Decision(next_node_ids) => {
                let max_child = next_node_ids
                    .iter()
                    .map(|next_node| self.longest_path(next_node.id.as_str()))
                    .max()
                    .unwrap_or(0);
                1 + max_child
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Airport security scenario
// ---------------------------------------------------------------------------

pub fn airport_security_scenario() -> GameTree {
    // Decision criteria are kept short and direct for small thinking
    // models. The PASS option is always listed first to avoid first-option
    // bias toward failure.
    let nodes = vec![
        GameNode {
            id: "START".into(),
            transcript: "Hello. Passport please.".into(),
            node_type: NodeType::Decision(vec![
                NextNode {
                    id: "PASSPORT_CHECK".into(),
                    description: "User proceeds to get his passport checked".into(),
                },
                NextNode {
                    id: "FAILED".into(),
                    description: "User failed the border inspection".into(),
                },
            ]),
            system_context: None,
        },
        GameNode {
            id: "PASSPORT_CHECK".into(),
            transcript: "Thank you. Let me take a look... Where are you travelling from today?"
                .into(),
            node_type: NodeType::Decision(vec![
                NextNode {
                    id: "QUESTION_PURPOSE".into(),
                    description: "User answered appropriately. Proceeding with the questioning."
                        .into(),
                },
                NextNode {
                    id: "FAILED".into(),
                    description: "User failed the questioning.".into(),
                },
            ]),
            system_context: Some(
                "EXAMPLES FOR PROPER RESPONSES:
                    - From Texas.
                    - I'm travelling from Frankfurt.
                    - From Atlanta, Georgia.
                "
                .into(),
            ),
        },
        GameNode {
            id: "QUESTION_PURPOSE".into(),
            transcript: "And what is the purpose of your visit?".into(),
            node_type: NodeType::Decision(vec![
                NextNode {
                    id: "LUGGAGE_CHECK".into(),
                    description: "User answered appropriately. Proceed to luggage check.".into(),
                },
                NextNode {
                    id: "FAILED".into(),
                    description: "User failed the questioning.".into(),
                },
            ]),
            system_context: None,
        },
        GameNode {
            id: "LUGGAGE_CHECK".into(),
            transcript: "Alright. Do you have anything to declare?".into(),
            node_type: NodeType::Decision(vec![
                NextNode {
                    id: "CLEARED".into(),
                    description: "traveller says nothing to declare or lists normal items.".into(),
                },
                NextNode {
                    id: "FAILED_CONTRABAND".into(),
                    description:
                        "traveller mentions illegal items, acts nervous, or is suspicious.".into(),
                },
            ]),
            system_context: Some("The guard asked about declarations.\n".into()),
        },
        // --- Terminal: success ---
        GameNode {
            id: "CLEARED".into(),
            transcript: "Everything checks out. Welcome, and enjoy your stay!".into(),
            node_type: NodeType::Terminal(true),
            system_context: None,
        },
        // --- Terminal: failures ---
        GameNode {
            id: "FAILED".into(),
            transcript: "I'm going to have to ask you to step aside. Security!".into(),
            node_type: NodeType::Terminal(false),
            system_context: None,
        },
    ];

    let mut map = HashMap::new();
    for node in nodes {
        map.insert(node.id.clone(), node);
    }

    GameTree {
        nodes: map,
        start_node_id: "START".into(),
    }
}
