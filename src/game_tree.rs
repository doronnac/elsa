use std::collections::HashMap;

/// A single node in the game's decision tree.
#[derive(Debug, Clone)]
pub struct GameNode {
    /// Unique identifier for this node (e.g. "START", "QUESTION_1", "FAILED").
    pub id: String,
    /// The line the guard says when entering this node.
    pub transcript: String,
    /// If true, the game ends at this node (win or lose).
    pub terminal: bool,
    /// Whether reaching this node counts as a success for the player.
    pub is_success: bool,
    /// IDs of possible next nodes the LLM can choose from.
    /// The POSITIVE/pass option should be listed first.
    pub next_node_ids: Vec<String>,
    /// Extra system-prompt context injected when the game reaches this node.
    /// Gives the LLM roleplay instructions specific to this stage.
    pub system_context: Option<String>,
}

/// The full scenario tree: a map of node-id -> GameNode.
#[derive(Debug, Clone)]
pub struct GameTree {
    pub nodes: HashMap<String, GameNode>,
    pub start_node_id: String,
}

impl GameTree {
    pub fn get(&self, id: &str) -> Option<&GameNode> {
        self.nodes.get(id)
    }

    pub fn start_node(&self) -> &GameNode {
        self.nodes
            .get(&self.start_node_id)
            .expect("start node must exist in tree")
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
        if node.terminal {
            return 0;
        }
        let max_child = node
            .next_node_ids
            .iter()
            .map(|id| self.longest_path(id))
            .max()
            .unwrap_or(0);
        1 + max_child
    }
}

// ---------------------------------------------------------------------------
// Airport security scenario
// ---------------------------------------------------------------------------

pub fn airport_security_scenario() -> GameTree {
    // NOTE: Decision criteria are kept short and direct for small thinking
    // models. The PASS option is always listed first to avoid first-option
    // bias toward failure.
    let nodes = vec![
        GameNode {
            id: "START".into(),
            transcript: "Hello. Passport please.".into(),
            terminal: false,
            is_success: false,
            // Pass option first
            next_node_ids: vec!["PASSPORT_CHECK".into(), "FAILED".into()],
            system_context: Some(
                "## CRITERIA: \n
                - **PASSPORT_CHECK**: Choose this if the traveller hands over the passport peacefully.
                - **FAILED**: Choose this if the traveller refuses, ignores, or answers inappropriately.
                ".into()
            ),
        },
        GameNode {
            id: "PASSPORT_CHECK".into(),
            transcript: "Thank you. Let me take a look... Where are you travelling from today?"
                .into(),
            terminal: false,
            is_success: false,
            next_node_ids: vec!["QUESTION_PURPOSE".into(), "FAILED".into()],
            system_context: Some(
                "## CRITERIA: \n
                - **QUESTION_PURPOSE**: traveller names a real place. It may be a country or city name. \n
                - **FAILED** traveller is evasive, vague, or contradictory. \n
                EXAMPLES FOR PROPER RESPONSES:
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
            terminal: false,
            is_success: false,
            next_node_ids: vec!["LUGGAGE_CHECK".into(), "FAILED_SUSPICIOUS".into()],
            system_context: Some(
                "The guard asked the purpose of the visit.\n\
                 LUGGAGE_CHECK = traveller gives a normal reason (tourism, business, family, etc.).\n\
                 FAILED_SUSPICIOUS = traveller refuses, mentions something illegal, or is evasive."
                    .into(),
            ),
        },
        GameNode {
            id: "LUGGAGE_CHECK".into(),
            transcript: "Alright. Do you have anything to declare?".into(),
            terminal: false,
            is_success: false,
            next_node_ids: vec!["CLEARED".into(), "FAILED_CONTRABAND".into()],
            system_context: Some(
                "The guard asked about declarations.\n\
                 CLEARED = traveller says nothing to declare or lists normal items.\n\
                 FAILED_CONTRABAND = traveller mentions illegal items, acts nervous, or is suspicious."
                    .into(),
            ),
        },
        // --- Terminal: success ---
        GameNode {
            id: "CLEARED".into(),
            transcript: "Everything checks out. Welcome, and enjoy your stay!".into(),
            terminal: true,
            is_success: true,
            next_node_ids: vec![],
            system_context: None,
        },
        // --- Terminal: failures ---
        GameNode {
            id: "FAILED".into(),
            transcript: "Sir/Ma'am, I'm going to have to ask you to step aside. Security!"
                .into(),
            terminal: true,
            is_success: false,
            next_node_ids: vec![],
            system_context: None,
        },
        GameNode {
            id: "FAILED_SUSPICIOUS".into(),
            transcript: "Your answers don't add up. Please follow me to secondary screening."
                .into(),
            terminal: true,
            is_success: false,
            next_node_ids: vec![],
            system_context: None,
        },
        GameNode {
            id: "FAILED_CONTRABAND".into(),
            transcript: "I'm going to need you to open your bags. Security has been notified."
                .into(),
            terminal: true,
            is_success: false,
            next_node_ids: vec![],
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
