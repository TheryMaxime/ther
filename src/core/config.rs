//! Central, runtime-switchable model configuration for the voice (Whisper) and
//! LLM (candle) engines.
//!
//! Resolution order (last wins): built-in **preset** -> optional `config.json`
//! -> environment variables. This keeps the historical env vars working while
//! giving a single place to switch model files/params.

use std::path::PathBuf;

use serde::Deserialize;

/// Fully-resolved configuration handed to the STT + LLM engines.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    // --- Voice / Whisper ---
    pub whisper_model_path: PathBuf,
    pub whisper_language: String,

    // --- LLM / candle ---
    pub llm_device_cpu: bool,
    pub llm_gguf_path: Option<PathBuf>,
    pub llm_gguf_repo: String,
    pub llm_gguf_file: String,
    pub llm_tokenizer_path: Option<PathBuf>,
    pub llm_tokenizer_repo: String,
    /// Local tokenizer filename (within the models dir) for this model.
    pub llm_tokenizer_file: String,

    // --- LLM generation / context tuning ---
    pub ctx_tokens: usize,
    pub max_new_tokens: usize,
    pub summary_tokens: usize,

    // --- Sampling (decoding) ---
    /// 0.0 => greedy (argmax); >0 enables temperature sampling.
    pub temperature: f64,
    /// Nucleus sampling threshold; <=0 or >=1 disables top-p.
    pub top_p: f64,
    /// Repetition penalty (1.0 = off); >1 discourages repeats/loops.
    pub repeat_penalty: f32,
    /// How many recent tokens the repetition penalty considers.
    pub repeat_last_n: usize,

    // --- Worker debounce / utterance gating ---
    pub debounce_ms: u64,
    pub min_chars: usize,
    pub min_new_chars: usize,
}

/// Optional `config.json` overlay. Every field is optional.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    whisper_preset: Option<String>,
    whisper_model_path: Option<String>,
    whisper_language: Option<String>,

    llm_preset: Option<String>,
    llm_device_cpu: Option<bool>,
    llm_gguf_path: Option<String>,
    llm_gguf_repo: Option<String>,
    llm_gguf_file: Option<String>,
    llm_tokenizer_path: Option<String>,
    llm_tokenizer_repo: Option<String>,
    llm_tokenizer_file: Option<String>,

    ctx_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    summary_tokens: Option<usize>,

    temperature: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,

    debounce_ms: Option<u64>,
    min_chars: Option<usize>,
    min_new_chars: Option<usize>,

    // --- Atlassian / Jira MCP ---
    atlassian_base_url: Option<String>,
    atlassian_email: Option<String>,
    atlassian_api_token: Option<String>,
    jira_default_project: Option<String>,
    jira_cloud_id: Option<String>,
    jira_mcp_mock: Option<bool>,
}

/// Atlassian Rovo MCP connection settings for the Jira module.
#[derive(Debug, Clone)]
pub struct JiraConfig {
    /// MCP endpoint, e.g. `https://mcp.atlassian.com/v1/mcp`.
    pub base_url: String,
    /// Atlassian account email (Basic-auth username).
    pub email: String,
    /// Atlassian API token (Basic-auth password).
    pub api_token: String,
    /// Optional default Jira project key used for new issues / JQL scoping.
    pub default_project: Option<String>,
    /// Optional pre-known cloudId (otherwise resolved at runtime).
    pub cloud_id: Option<String>,
    /// When true, the MCP client returns canned data instead of hitting the network.
    pub mock: bool,
}

impl JiraConfig {
    /// Resolve Atlassian settings from `config.json` then environment overrides.
    pub fn load() -> Self {
        let file = load_file_config();

        let base_url = env_string("ATLASSIAN_BASE_URL")
            .or(file.atlassian_base_url.clone())
            .unwrap_or_else(|| "https://mcp.atlassian.com/v1/mcp".to_string());
        let email = env_string("ATLASSIAN_EMAIL")
            .or(file.atlassian_email.clone())
            .unwrap_or_default();
        let api_token = env_string("ATLASSIAN_API_TOKEN")
            .or(file.atlassian_api_token.clone())
            .unwrap_or_default();
        let default_project =
            env_string("JIRA_DEFAULT_PROJECT").or(file.jira_default_project.clone());
        let cloud_id = env_string("JIRA_CLOUD_ID").or(file.jira_cloud_id.clone());
        let mock = std::env::var("JIRA_MCP_MOCK")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .or(file.jira_mcp_mock)
            .unwrap_or(false);

        Self {
            base_url,
            email,
            api_token,
            default_project,
            cloud_id,
            mock,
        }
    }

    /// True when real credentials are present (or mock mode is on).
    pub fn is_usable(&self) -> bool {
        self.mock || (!self.email.is_empty() && !self.api_token.is_empty())
    }
}

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("LLM_MODEL_DIR").unwrap_or_else(|_| "models".to_string()))
}

/// Apply a named Whisper preset (model file within the models dir).
fn whisper_preset_path(name: &str) -> PathBuf {
    let file = match name {
        "base.en" => "ggml-base.en.bin",
        "small" => "ggml-small.bin",
        "small.en" => "ggml-small.en.bin",
        "medium" => "ggml-medium.bin",
        // "base" and anything unknown fall back to base.
        _ => "ggml-base.bin",
    };
    models_dir().join(file)
}

/// A named LLM preset: where to get the weights + tokenizer, and the local
/// tokenizer filename produced by `scripts/download-llm.sh`.
#[derive(Debug, Clone, Copy)]
pub struct LlmPreset {
    pub gguf_repo: &'static str,
    pub gguf_file: &'static str,
    pub tokenizer_repo: &'static str,
    pub tokenizer_file: &'static str,
}

/// Ministral-3B: small/fast, weak instruction-following (default, backward compat).
const PRESET_MINISTRAL_3B: LlmPreset = LlmPreset {
    gguf_repo: "QuantFactory/Ministral-3b-instruct-GGUF",
    gguf_file: "Ministral-3b-instruct.Q4_K_M.gguf",
    tokenizer_repo: "ministral/Ministral-3b-instruct",
    tokenizer_file: "ministral-tokenizer.json",
};

/// Mistral-7B-Instruct v0.2: much stronger, same `[INST]` prompt format.
const PRESET_MISTRAL_7B: LlmPreset = LlmPreset {
    gguf_repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    tokenizer_repo: "mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_file: "mistral-7b-instruct-v0.2-tokenizer.json",
};

/// Resolve a named LLM preset (unknown names fall back to Ministral-3B).
pub fn llm_preset(name: &str) -> LlmPreset {
    match name.trim().to_ascii_lowercase().as_str() {
        "mistral-7b" | "mistral-7b-instruct" | "mistral" => PRESET_MISTRAL_7B,
        _ => PRESET_MINISTRAL_3B,
    }
}

fn load_file_config() -> FileConfig {
    let path = std::env::var("AI4JIRA_CONFIG").unwrap_or_else(|_| "config.json".to_string());
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            eprintln!("[config] failed to parse {path}: {e}; ignoring");
            FileConfig::default()
        }),
        Err(_) => FileConfig::default(),
    }
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn env_f32(key: &str) -> Option<f32> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

impl ModelConfig {
    /// Resolve the effective configuration from presets, `config.json`, and env.
    pub fn load() -> Self {
        let file = load_file_config();

        // --- Whisper ---
        let whisper_preset = env_string("WHISPER_PRESET")
            .or(file.whisper_preset.clone())
            .unwrap_or_else(|| "base".to_string());
        let whisper_model_path = env_string("WHISPER_MODEL_PATH")
            .or(file.whisper_model_path.clone())
            .map(PathBuf::from)
            .unwrap_or_else(|| whisper_preset_path(&whisper_preset));
        let whisper_language = env_string("WHISPER_LANG")
            .or(file.whisper_language.clone())
            .unwrap_or_else(|| "en".to_string());

        // --- LLM model files (preset provides defaults) ---
        let llm_preset_name = env_string("LLM_PRESET")
            .or(file.llm_preset.clone())
            .unwrap_or_else(|| "ministral-3b".to_string());
        let preset = llm_preset(&llm_preset_name);

        let llm_gguf_path = env_string("LLM_GGUF_PATH")
            .or(file.llm_gguf_path.clone())
            .map(PathBuf::from);
        let llm_gguf_repo = env_string("LLM_GGUF_REPO")
            .or(file.llm_gguf_repo.clone())
            .unwrap_or_else(|| preset.gguf_repo.to_string());
        let llm_gguf_file = env_string("LLM_GGUF_FILE")
            .or(file.llm_gguf_file.clone())
            .unwrap_or_else(|| preset.gguf_file.to_string());
        let llm_tokenizer_path = env_string("LLM_TOKENIZER_PATH")
            .or(file.llm_tokenizer_path.clone())
            .map(PathBuf::from);
        let llm_tokenizer_repo = env_string("LLM_TOKENIZER_REPO")
            .or(file.llm_tokenizer_repo.clone())
            .unwrap_or_else(|| preset.tokenizer_repo.to_string());
        let llm_tokenizer_file = env_string("LLM_TOKENIZER_FILE")
            .or(file.llm_tokenizer_file.clone())
            .unwrap_or_else(|| preset.tokenizer_file.to_string());

        let llm_device_cpu = std::env::var("LLM_DEVICE")
            .ok()
            .map(|v| v == "cpu")
            .or(file.llm_device_cpu)
            .unwrap_or(false);

        // --- Tuning ---
        let ctx_tokens = env_usize("LLM_CTX_TOKENS")
            .or(file.ctx_tokens)
            .unwrap_or(3900);
        let max_new_tokens = env_usize("LLM_MAX_NEW_TOKENS")
            .or(file.max_new_tokens)
            .unwrap_or(160);
        let summary_tokens = env_usize("LLM_SUMMARY_TOKENS")
            .or(file.summary_tokens)
            .unwrap_or(512);

        // --- Sampling ---
        let temperature = env_f64("LLM_TEMPERATURE")
            .or(file.temperature)
            .unwrap_or(0.3);
        let top_p = env_f64("LLM_TOP_P").or(file.top_p).unwrap_or(0.9);
        let repeat_penalty = env_f32("LLM_REPEAT_PENALTY")
            .or(file.repeat_penalty)
            .unwrap_or(1.15);
        let repeat_last_n = env_usize("LLM_REPEAT_LAST_N")
            .or(file.repeat_last_n)
            .unwrap_or(64);

        let debounce_ms = env_u64("LLM_DEBOUNCE_MS")
            .or(file.debounce_ms)
            .unwrap_or(3000);
        let min_chars = env_usize("LLM_MIN_CHARS").or(file.min_chars).unwrap_or(40);
        let min_new_chars = env_usize("LLM_MIN_NEW_CHARS")
            .or(file.min_new_chars)
            .unwrap_or(160);

        Self {
            whisper_model_path,
            whisper_language,
            llm_device_cpu,
            llm_gguf_path,
            llm_gguf_repo,
            llm_gguf_file,
            llm_tokenizer_path,
            llm_tokenizer_repo,
            llm_tokenizer_file,
            ctx_tokens,
            max_new_tokens,
            summary_tokens,
            temperature,
            top_p,
            repeat_penalty,
            repeat_last_n,
            debounce_ms,
            min_chars,
            min_new_chars,
        }
    }
}
