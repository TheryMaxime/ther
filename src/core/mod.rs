//! UI-agnostic core services shared by all modules: model configuration, the
//! speech-to-text recorder, and the embedded LLM engine/worker.
//!
//! Nothing in `core` depends on a concrete UI. Instead, modules pass in an
//! [`EventSender`] that core services use to report transcript / response /
//! status [`CoreEvent`]s, and drive whatever UI they ship from the matching
//! receiver.

pub mod bus;
pub mod config;
pub mod llm;
pub mod mcp;
pub mod stt;

pub use bus::{CoreEvent, EventSender};

/// Domain context a module injects into the LLM so its answers fit the module's
/// purpose (e.g. Jira task extraction).
#[derive(Debug, Clone, Default)]
pub struct LlmContext {
    /// High-level instruction describing how the assistant should behave.
    pub system_prompt: String,
    /// Extra domain schema / struct description appended to the system prompt.
    pub context: String,
}

impl LlmContext {
    /// Combined system prompt actually fed to the model.
    pub fn effective_prompt(&self) -> String {
        let base = self.system_prompt.trim();
        let ctx = self.context.trim();
        if ctx.is_empty() {
            base.to_string()
        } else {
            format!("{base}\n\n{ctx}")
        }
    }
}
