//! Pluggable application "modules". Each module defines the app's *finality*:
//! it ships its own Dioxus screen plus domain logic (system prompt, LLM context,
//! output parsing and actions). Exactly one module is selected at **compile
//! time** via a Cargo feature.

use crate::core::LlmContext;

#[cfg(feature = "jira")]
pub mod jira;

/// Contract implemented by every module.
pub trait Module {
    /// Stable identifier, e.g. `"jira"`.
    fn id(&self) -> &'static str;
    /// Human-readable window title.
    fn title(&self) -> &'static str;
    /// Domain context injected into the LLM so answers fit this module's purpose.
    fn llm_context(&self) -> LlmContext;
    /// Build the module's UI, wire it to the core services, and run the event loop.
    fn run(&self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Run the module selected by the active Cargo feature.
pub fn run_active() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "jira")]
    {
        let module = jira::JiraModule;
        eprintln!("[module] starting '{}' ({})", module.id(), module.title());
        return module.run();
    }

    #[allow(unreachable_code)]
    {
        eprintln!("No module feature enabled. Build with e.g. --features jira");
        Ok(())
    }
}
