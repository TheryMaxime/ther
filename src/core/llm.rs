use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama::{ModelWeights, MAX_SEQ_LEN};
use candle_transformers::utils::apply_repeat_penalty;
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;
use tokenizers::Tokenizer;

use crate::core::config::ModelConfig;
use crate::core::{Callback, LlmContext};

/// Default assistant instruction, used when a module supplies no system prompt.
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a concise assistant listening to an ongoing spoken conversation. \
Use the running summary of earlier discussion together with the most recent \
transcript to understand the full context, then respond helpfully in one or two \
short sentences.";

/// Prompt used to fold older transcript into a compact running summary so the
/// model can effectively "remember" a much longer part of the discussion than
/// fits in the context window.
const SUMMARIZER_PROMPT: &str =
    "You are maintaining running notes of a spoken conversation. Update the summary \
to incorporate the new transcript excerpt. Keep it concise and factual (a few \
sentences at most); preserve decisions, action items, names, dates and numbers.";

/// Extra tokens reserved for prompt scaffolding (INST markers, system prompt, labels).
const SCAFFOLD_RESERVE: usize = 96;

/// Messages accepted by the background LLM worker.
///
/// The worker multiplexes two jobs on the single embedded engine: streaming
/// answers to live transcript snapshots, and one-off structured completions
/// (e.g. Jira proposal extraction) requested on demand by a module.
pub enum LlmMessage {
    /// A new transcript snapshot to (eventually) answer.
    Transcript(String),
    /// A one-off completion from a fully-built prompt. The result is delivered
    /// via `reply`; the live-answer state (summary, debounce) is left untouched.
    Complete {
        prompt: String,
        max_new_tokens: usize,
        reply: Callback,
    },
}

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("LLM_MODEL_DIR").unwrap_or_else(|_| "models".to_string()))
}

/// Local tokenizer filename produced by scripts/download-llm.sh.
/// Legacy local tokenizer filename (pre-preset). Kept as a fallback so existing
/// downloads keep working.
const LEGACY_TOKENIZER_FILE: &str = "ministral-tokenizer.json";

/// Resolve the GGUF weights path: explicit config path, then a local file in the
/// models dir (downloaded via scripts/download-llm.sh), then hf-hub download.
fn resolve_gguf_path(cfg: &ModelConfig) -> Result<PathBuf> {
    if let Some(p) = &cfg.llm_gguf_path {
        return Ok(p.clone());
    }
    let local = models_dir().join(&cfg.llm_gguf_file);
    if local.exists() {
        return Ok(local);
    }
    let api = Api::new().context("failed to init hf-hub api")?;
    api.model(cfg.llm_gguf_repo.clone())
        .get(&cfg.llm_gguf_file)
        .with_context(|| {
            format!(
                "failed to fetch {}/{} via hf-hub. \
Behind a proxy? Pre-download with scripts/download-llm.sh",
                cfg.llm_gguf_repo, cfg.llm_gguf_file
            )
        })
}

/// Resolve the tokenizer.json path: config path, then local models dir, then hf-hub.
fn resolve_tokenizer_path(cfg: &ModelConfig) -> Result<PathBuf> {
    if let Some(p) = &cfg.llm_tokenizer_path {
        return Ok(p.clone());
    }
    let local = models_dir().join(&cfg.llm_tokenizer_file);
    if local.exists() {
        return Ok(local);
    }
    let legacy = models_dir().join(LEGACY_TOKENIZER_FILE);
    if legacy.exists() {
        return Ok(legacy);
    }
    let api = Api::new().context("failed to init hf-hub api")?;
    api.model(cfg.llm_tokenizer_repo.clone())
        .get("tokenizer.json")
        .with_context(|| {
            format!(
                "failed to fetch tokenizer from {} via hf-hub. \
Behind a proxy? Pre-download with scripts/download-llm.sh",
                cfg.llm_tokenizer_repo
            )
        })
}

fn select_device(cfg: &ModelConfig) -> Device {
    if cfg.llm_device_cpu {
        return Device::Cpu;
    }
    match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Metal device unavailable ({e}); falling back to CPU");
            Device::Cpu
        }
    }
}

/// Embedded quantized LLM (e.g. Ministral-3B GGUF) used to answer the live
/// transcript. Context sizing and the system prompt are configurable so the
/// engine can serve any module.
pub struct LlmEngine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    eos_token: u32,
    system_prompt: String,
    ctx_tokens: usize,
    max_new_tokens: usize,
    summary_tokens: usize,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl LlmEngine {
    /// Download (first run) and load the GGUF weights + tokenizer per `cfg`.
    pub fn load(cfg: &ModelConfig) -> Result<Self> {
        let device = select_device(cfg);

        let model_path = resolve_gguf_path(cfg)?;
        let tokenizer_path = resolve_tokenizer_path(cfg)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        let mut tokenizer = tokenizer;
        // Ensure no fixed-length padding/truncation is applied to our prompts.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("failed to disable truncation: {e}"))?;

        let mut file = std::fs::File::open(&model_path)
            .with_context(|| format!("failed to open {}", model_path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(&model_path))
            .context("failed to read gguf content")?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .context("failed to build model from gguf")?;

        let eos_token = tokenizer
            .get_vocab(true)
            .get("</s>")
            .copied()
            .context("tokenizer missing </s> token")?;

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            ctx_tokens: cfg.ctx_tokens.min(MAX_SEQ_LEN),
            max_new_tokens: cfg.max_new_tokens,
            summary_tokens: cfg.summary_tokens,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            repeat_penalty: cfg.repeat_penalty,
            repeat_last_n: cfg.repeat_last_n,
        })
    }

    /// Override the assistant instruction (used by modules to inject domain context).
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        let prompt = prompt.into();
        if !prompt.trim().is_empty() {
            self.system_prompt = prompt;
        }
    }

    /// Token budget for the verbatim, most-recent portion of the transcript.
    fn recent_budget(&self) -> usize {
        self.ctx_tokens
            .saturating_sub(self.max_new_tokens + self.summary_tokens + SCAFFOLD_RESERVE)
            .max(256)
    }

    /// Final safety budget for the assembled answer prompt.
    fn answer_prompt_budget(&self) -> usize {
        self.ctx_tokens.saturating_sub(self.max_new_tokens + 10)
    }

    /// Encode text into token ids (no special tokens added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))
    }

    /// Core generation from an already-tokenized prompt. Uses a fresh context
    /// (KV cache cleared) so each call is independent. Decoding follows the
    /// engine's sampling config (greedy when `temperature == 0`) and applies a
    /// repetition penalty to curb the degenerate loops small models fall into.
    fn run_tokens(&mut self, mut prompt_tokens: Vec<u32>, budget: usize, max_new: usize) -> Result<String> {
        self.model.clear_kv_cache();

        // Final safety: keep the prompt within the model's context window,
        // preserving the most recent (tail) tokens.
        if prompt_tokens.len() > budget {
            prompt_tokens = prompt_tokens[prompt_tokens.len() - budget..].to_vec();
        }

        let mut logits_processor = LogitsProcessor::from_sampling(42, self.sampling());
        let mut all_tokens = prompt_tokens.clone();

        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?.squeeze(0)?;
        let logits = self.apply_penalty(&logits, &all_tokens)?;
        let mut next_token = logits_processor.sample(&logits)?;

        let mut output_tokens: Vec<u32> = Vec::new();
        let mut index_pos = prompt_tokens.len();

        if std::env::var("LLM_DEBUG").is_ok() {
            eprintln!(
                "[llm] prompt_tokens={} max_new={} eos_id={} temp={} top_p={} rep={} first_sampled={}",
                prompt_tokens.len(),
                max_new,
                self.eos_token,
                self.temperature,
                self.top_p,
                self.repeat_penalty,
                next_token
            );
        }

        for _ in 0..max_new {
            if next_token == self.eos_token {
                break;
            }
            output_tokens.push(next_token);
            all_tokens.push(next_token);

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, index_pos)?.squeeze(0)?;
            let logits = self.apply_penalty(&logits, &all_tokens)?;
            next_token = logits_processor.sample(&logits)?;
            index_pos += 1;
        }

        Ok(self.decode(&output_tokens)?.trim().to_string())
    }

    /// Build the sampling strategy from the engine's decoding config.
    fn sampling(&self) -> Sampling {
        if self.temperature <= 0.0 {
            Sampling::ArgMax
        } else if self.top_p > 0.0 && self.top_p < 1.0 {
            Sampling::TopP {
                p: self.top_p,
                temperature: self.temperature,
            }
        } else {
            Sampling::All {
                temperature: self.temperature,
            }
        }
    }

    /// Apply the repetition penalty over the last `repeat_last_n` tokens.
    fn apply_penalty(&self, logits: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        if (self.repeat_penalty - 1.0).abs() < f32::EPSILON {
            return Ok(logits.clone());
        }
        let start = tokens.len().saturating_sub(self.repeat_last_n);
        Ok(apply_repeat_penalty(logits, self.repeat_penalty, &tokens[start..])?)
    }

    /// Fold an older transcript excerpt into the running `summary` so the model
    /// retains a much longer part of the discussion than fits verbatim.
    pub fn summarize(&mut self, prev_summary: &str, excerpt: &str) -> Result<String> {
        let prev = if prev_summary.trim().is_empty() {
            "(none yet)"
        } else {
            prev_summary
        };
        let prompt = format!(
            "[INST] {SUMMARIZER_PROMPT}\n\nCurrent summary:\n{prev}\n\n\
New transcript excerpt:\n{excerpt}\n\nUpdated summary: [/INST]"
        );
        let tokens = self.encode(&prompt)?;
        let budget = self.ctx_tokens.saturating_sub(self.summary_tokens + 10);
        self.run_tokens(tokens, budget, self.summary_tokens)
    }

    /// Generate a short answer from the running `summary` plus the verbatim
    /// `recent` transcript.
    pub fn answer(&mut self, summary: &str, recent: &str) -> Result<String> {
        let system_prompt = &self.system_prompt;
        let prompt = if summary.trim().is_empty() {
            format!("[INST] {system_prompt}\n\nRecent transcript:\n{recent} [/INST]")
        } else {
            format!(
                "[INST] {system_prompt}\n\nSummary of earlier discussion:\n{summary}\n\n\
Recent transcript:\n{recent} [/INST]"
            )
        };
        let tokens = self.encode(&prompt)?;
        let budget = self.answer_prompt_budget();
        self.run_tokens(tokens, budget, self.max_new_tokens)
    }

    /// One-shot convenience: answer for a full transcript, summarizing any part
    /// that overflows the recent-token window. Stateless (no persistent memory).
    pub fn generate(&mut self, transcript: &str) -> Result<String> {        let ids = self.encode(transcript)?;
        let recent_budget = self.recent_budget();

        if ids.len() <= recent_budget {
            return self.answer("", transcript);
        }

        let split = ids.len() - recent_budget;
        let older = self.decode(&ids[..split])?;
        let recent = self.decode(&ids[split..])?;
        let summary = self.summarize("", &older)?;
        self.answer(&summary, &recent)
    }

    /// Generic single-shot completion from a fully-built prompt. Used for
    /// structured extraction (e.g. Jira proposals) where the caller controls the
    /// entire prompt and expects the raw model output back.
    pub fn complete(&mut self, prompt: &str, max_new: usize) -> Result<String> {
        let tokens = self.encode(prompt)?;
        let budget = self.ctx_tokens.saturating_sub(max_new + 10);
        self.run_tokens(tokens, budget, max_new)
    }
}

/// Spawn the background LLM worker.
///
/// It loads the model once, then answers transcript snapshots received on `rx`.
///
/// To make the model comprehend better it does two things:
///   1. **Waits for fuller utterances** — snapshots are debounced: it only
///      answers once the transcript has been quiet for `LLM_DEBOUNCE_MS`, or
///      enough new content (`LLM_MIN_NEW_CHARS`) has accumulated, and never
///      before `LLM_MIN_CHARS` of transcript exist.
///   2. **Takes a longer part of the discussion** — it keeps a persistent
///      rolling `summary` of the older transcript so the model effectively sees
///      the whole conversation, not just the tail that fits in the window.
pub fn spawn_worker(
    rx: Receiver<LlmMessage>,
    cfg: ModelConfig,
    ctx: LlmContext,
    on_response: Callback,
    on_status: Callback,
) {
    std::thread::spawn(move || {
        on_status("Loading LLM (first run may download the model)...".to_string());

        let mut engine = match LlmEngine::load(&cfg) {
            Ok(e) => e,
            Err(e) => {
                on_status(format!("LLM unavailable: {e}"));
                return;
            }
        };
        engine.set_system_prompt(ctx.effective_prompt());
        on_status("LLM ready.".to_string());

        let debounce = Duration::from_millis(cfg.debounce_ms);
        let min_chars = cfg.min_chars;
        let min_new_chars = cfg.min_new_chars;

        // Persistent rolling memory across the whole session.
        let mut summary = String::new();
        let mut summarized_prefix_tokens = 0usize;
        let mut answered_len = 0usize;

        // The most recent transcript snapshot awaiting an answer, if any.
        let mut latest: Option<String> = None;

        loop {
            // Debounce only while a *long-enough* transcript is pending; otherwise
            // block for the next message so we don't spin on short fragments.
            let answerable = latest
                .as_ref()
                .map_or(false, |t| t.trim().len() >= min_chars);

            let received = if answerable {
                match rx.recv_timeout(debounce) {
                    Ok(msg) => Some(msg),
                    Err(RecvTimeoutError::Timeout) => None, // quiet -> answer now
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            } else {
                match rx.recv() {
                    Ok(msg) => Some(msg),
                    Err(_) => return,
                }
            };

            match received {
                Some(LlmMessage::Complete {
                    prompt,
                    max_new_tokens,
                    reply,
                }) => {
                    on_status("Analyzing transcript...".to_string());
                    match engine.complete(&prompt, max_new_tokens) {
                        Ok(out) => reply(out),
                        Err(e) => {
                            on_status(format!("LLM error: {e}"));
                            reply(String::new());
                        }
                    }
                    on_status("LLM ready.".to_string());
                }
                Some(LlmMessage::Transcript(t)) => {
                    let enough_new =
                        t.trim().len() >= min_chars && t.len().saturating_sub(answered_len) >= min_new_chars;
                    latest = Some(t);
                    if enough_new {
                        answer_now(
                            &mut engine,
                            &mut latest,
                            &mut summary,
                            &mut summarized_prefix_tokens,
                            &mut answered_len,
                            &on_response,
                            &on_status,
                        );
                    }
                }
                None => {
                    // Debounce timeout with an answerable transcript pending.
                    answer_now(
                        &mut engine,
                        &mut latest,
                        &mut summary,
                        &mut summarized_prefix_tokens,
                        &mut answered_len,
                        &on_response,
                        &on_status,
                    );
                }
            }
        }
    });
}

/// Answer the pending transcript, update rolling memory, and clear `latest` so
/// the worker blocks for the next snapshot.
#[allow(clippy::too_many_arguments)]
fn answer_now(
    engine: &mut LlmEngine,
    latest: &mut Option<String>,
    summary: &mut String,
    summarized_prefix_tokens: &mut usize,
    answered_len: &mut usize,
    on_response: &Callback,
    on_status: &Callback,
) {
    let Some(text) = latest.clone() else {
        return;
    };

    on_status("Assistant thinking...".to_string());
    match answer_with_memory(engine, &text, summary, summarized_prefix_tokens) {
        Ok(response) => {
            *answered_len = text.len();
            on_response(response);
        }
        Err(e) => on_status(format!("LLM error: {e}")),
    }
    // Consumed; wait for the next transcript before answering again.
    *latest = None;
}

/// Answer the current transcript while maintaining a persistent rolling summary.
///
/// The verbatim recent window is capped to the engine's `recent_budget`; anything
/// older that hasn't been summarized yet is folded into `summary`, so the model
/// keeps access to a much longer part of the discussion than fits in context.
fn answer_with_memory(
    engine: &mut LlmEngine,
    transcript: &str,
    summary: &mut String,
    summarized_prefix_tokens: &mut usize,
) -> Result<String> {
    let ids = engine.encode(transcript)?;
    let recent_budget = engine.recent_budget();

    if ids.len() <= recent_budget {
        return engine.answer(summary, transcript);
    }

    let recent_start = ids.len() - recent_budget;

    // Fold any not-yet-summarized older tokens into the running summary.
    if recent_start > *summarized_prefix_tokens {
        let excerpt = engine.decode(&ids[*summarized_prefix_tokens..recent_start])?;
        if !excerpt.trim().is_empty() {
            *summary = engine.summarize(summary, &excerpt)?;
            *summarized_prefix_tokens = recent_start;
        }
    }

    let recent = engine.decode(&ids[recent_start..])?;
    engine.answer(summary, &recent)
}
