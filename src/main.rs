mod core;
mod modules;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional headless check of the embedded LLM path (no UI/mic needed):
    //   LLM_SELFTEST=1 cargo run
    // Override the input with LLM_SELFTEST_TEXT to exercise the summary path.
    if std::env::var("LLM_SELFTEST").is_ok() {
        let cfg = core::config::ModelConfig::load();
        let transcript = std::env::var("LLM_SELFTEST_TEXT").unwrap_or_else(|_| {
            "We need to add JWT validation to the auth endpoint before Friday.".to_string()
        });
        match core::llm::LlmEngine::load(&cfg).and_then(|mut e| e.generate(&transcript)) {
            Ok(answer) => println!("LLM self-test answer:\n{answer}"),
            Err(e) => eprintln!("LLM self-test failed: {e:?}"),
        }
        return Ok(());
    }

    // Optional headless check of the Jira proposal-extraction path.
    #[cfg(feature = "jira")]
    if std::env::var("JIRA_PROPOSAL_SELFTEST").is_ok() {
        modules::jira::proposal_selftest();
        return Ok(());
    }

    modules::run_active()
}
