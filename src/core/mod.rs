//! UI-agnostic core services shared by all modules: model configuration, the
//! speech-to-text recorder, and the embedded LLM engine/worker.
//!
//! Nothing in `core` depends on a concrete Slint window. Instead, modules pass
//! in [`Callback`]s that receive transcript / response / status updates and are
//! free to drive whatever UI they ship.

pub mod config;
pub mod llm;
pub mod mcp;
pub mod stt;

/// A thread-safe sink for a string update (transcript, response, or status).
pub type Callback = Box<dyn Fn(String) + Send + 'static>;

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
