//! JIRA module: turn a live meeting transcript into **reviewable** Jira story /
//! epic changes.
//!
//! Ships its own Dioxus screen (`ui.rs`) and owns all Jira-specific logic:
//! the LLM system prompt / context, detecting which issues are under discussion,
//! prompting the embedded LLM for structured change *proposals*, and applying
//! approved proposals to Jira through the Atlassian MCP server. Nothing is
//! written to Jira without explicit per-proposal approval.
//!
//! Proposal extraction is seeded from the assistant's own running-notes
//! *answer* to the transcript (not the raw transcript itself): the LLM has
//! already condensed the discussion once to produce that answer, so
//! extraction re-analyzes that condensed text instead of re-prompting the
//! model with the full raw dialogue a second time. The raw transcript is
//! still used locally for cheap regex-based issue-key detection.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use tokio::runtime::Runtime;

use super::Module;
use crate::core::config::{JiraConfig, ModelConfig};
use crate::core::llm::{self, LlmRequest};
use crate::core::stt::Recorder;
use crate::core::{ContextStore, EventSender, LlmContext};

use dioxus::desktop::{Config, WindowBuilder};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

mod jira_client;
mod proposal;
mod ui;

use jira_client::{created_key, JiraClient};
use ui::{AppJira, ProposalView, Update};
use proposal::{
    build_extraction_prompt, detect_issue_keys, normalized_summary, parse_proposals,
    similarity_score, Proposal, ProposalAction,
};

/// [`ContextStore`] namespace this module stores its data under.
const MODULE_ID: &str = "jira";
/// Collection name for reviewable proposals within the store.
const PROPOSALS_COLLECTION: &str = "proposals";
/// Minimum token-overlap similarity to treat two proposals as "the same"
/// when neither `target_key` nor the normalized summary match exactly.
const SIMILARITY_MATCH_THRESHOLD: f32 = 0.6;

/// Assistant instruction for the Jira use-case (live running notes).
const JIRA_SYSTEM_PROMPT: &str =
    "You are an assistant listening to an engineering meeting about Jira user \
stories and epics. Track the discussion and keep concise running notes of the \
concrete changes (new stories, edits to existing issues) the team is agreeing on.";

/// Domain schema injected as LLM context so answers match the Jira story shape.
const JIRA_CONTEXT: &str =
    "Work is organised as Jira issues: Epics contain Stories; a Story has a \
`summary` (imperative title), a `description`, and `acceptance criteria`.";

/// Max tokens for a structured proposal-extraction completion.
const PROPOSAL_MAX_TOKENS: usize = 640;

/// A proposal plus its review state, held in Rust as the source of truth.
///
/// Also mirrored into the module's [`ContextStore`] namespace (see
/// [`Backend::sync_context`]) so it can be fetched generically, the same way
/// any other module's cached context can.
#[derive(Clone, Serialize, Deserialize)]
struct ProposalItem {
    id: i32,
    proposal: Proposal,
    /// "pending" | "approved" | "rejected" | "applied" | "error".
    status: String,
    result: String,
}

/// The Jira module entry point.
pub struct JiraModule;

impl Module for JiraModule {
    fn id(&self) -> &'static str {
        "jira"
    }

    fn title(&self) -> &'static str {
        "Meeting to Jira"
    }

    fn llm_context(&self) -> LlmContext {
        LlmContext {
            system_prompt: JIRA_SYSTEM_PROMPT.to_string(),
            context: JIRA_CONTEXT.to_string(),
        }
    }

    fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        run_app(self.llm_context())
    }
}

/// Headless check of the proposal-extraction path (no UI/mic/network):
///   JIRA_PROPOSAL_SELFTEST=1 cargo run
/// Loads the LLM, first generates its running-notes answer for a sample
/// transcript (mirroring the live pipeline), then runs the extraction prompt
/// over those notes, and prints the raw output plus the defensively-parsed
/// proposals.
pub fn proposal_selftest() {
    let transcript = std::env::var("JIRA_PROPOSAL_SELFTEST_TEXT").unwrap_or_else(|_| {
        "Let's create a new story for password reset via email; users get a link \
that expires in one hour. Also we need to update PROJ-42, the onboarding epic, to \
drop the SMS step."
            .to_string()
    });

    let cfg = ModelConfig::load();
    match crate::core::llm::LlmEngine::load(&cfg) {
        Ok(mut engine) => {
            let notes = match engine.generate(&transcript) {
                Ok(notes) => {
                    println!("--- ASSISTANT NOTES ---\n{notes}\n------------------------");
                    notes
                }
                Err(e) => {
                    eprintln!("proposal self-test failed to generate notes: {e:?}");
                    return;
                }
            };
            let prompt = build_extraction_prompt(&notes, "", "PROJ");
            match engine.complete(&prompt, PROPOSAL_MAX_TOKENS) {
                Ok(raw) => {
                    println!("--- RAW MODEL OUTPUT ---\n{raw}\n------------------------");
                    let props = parse_proposals(&raw);
                    println!("Parsed {} proposal(s):", props.len());
                    for p in &props {
                        println!(
                            "  [{}] type={} target='{}' :: {} | {}",
                            p.action.as_str(),
                            p.issue_type,
                            p.target_key,
                            p.summary,
                            p.description
                        );
                    }
                }
                Err(e) => eprintln!("proposal self-test failed: {e:?}"),
            }
        }
        Err(e) => eprintln!("proposal self-test failed: {e:?}"),
    }
}

// --- Backend: shared services + the UI's action surface --------------------

/// Handle passed to the Dioxus UI (via context). Holds every shared service and
/// exposes the actions the UI can trigger. All fields are `Arc`/`Send` so the
/// handle is cheaply cloneable and thread-safe; background work is pushed back
/// to the UI as [`Update`]s over `update_tx`.
#[derive(Clone)]
pub struct Backend {
    cfg: ModelConfig,
    jira_cfg: Arc<JiraConfig>,
    tokio_rt: Arc<Runtime>,
    jira_client: Arc<tokio::sync::Mutex<Option<JiraClient>>>,
    items: Arc<Mutex<Vec<ProposalItem>>>,
    next_id: Arc<AtomicI32>,
    /// Generic in-memory cache mirroring `items`, exposed via the same
    /// [`ContextStore`] API any module can use to fetch/update previously
    /// generated context (see [`Backend::sync_context`]).
    context: ContextStore,
    /// Typed request channel into the embedded LLM worker.
    llm_tx: mpsc::Sender<LlmRequest>,
    /// Shared event sink handed to core services (STT, LLM) for status /
    /// transcript / response updates.
    events: EventSender,
    /// UI-facing update channel (core events are adapted into it, plus the
    /// Jira-specific proposal/analyze updates the Backend emits directly).
    update_tx: UnboundedSender<Update>,
    update_rx: Arc<Mutex<Option<UnboundedReceiver<Update>>>>,
}

impl Backend {
    fn emit(&self, update: Update) {
        let _ = self.update_tx.send(update);
    }

    /// Take the single UI-update receiver (consumed once by the UI's drain task).
    pub fn take_update_rx(&self) -> Option<UnboundedReceiver<Update>> {
        self.update_rx.lock().unwrap().take()
    }

    /// Character growth in the assistant's running notes that auto-triggers
    /// analysis (extraction reanalyzes the notes, not the raw transcript).
    pub fn analyze_threshold(&self) -> usize {
        self.cfg.analyze_min_new_chars
    }

    /// Push the current proposal list into the UI.
    fn render_proposals(&self) {
        let snapshot: Vec<ProposalView> =
            self.items.lock().unwrap().iter().map(to_proposal_view).collect();
        self.emit(Update::Proposals(snapshot));
    }

    /// Mirror the current `items` into the generic [`ContextStore`] so they
    /// can be fetched later the same way any module's cached context can
    /// (`context.list::<ProposalItem>("jira", "proposals")`).
    fn sync_context(&self) {
        let guard = self.items.lock().unwrap();
        for item in guard.iter() {
            let _ = self.context.put(MODULE_ID, PROPOSALS_COLLECTION, item.id.to_string(), item);
        }
    }

    /// Start microphone capture + live transcription. The returned [`Recorder`]
    /// is `!Send` and must be owned by the (main-thread) UI.
    pub fn start_recording(&self) -> Result<Recorder, String> {
        Recorder::start(
            self.cfg.whisper_model_path.clone(),
            self.cfg.whisper_language.clone(),
            self.events.clone(),
            self.llm_tx.clone(),
        )
    }

    /// Handle a typed message: append it to the transcript (so live answers and
    /// `Analyze` treat it like spoken input) and forward it to the LLM worker.
    pub fn send_message(&self, current_transcript: String, input: String) {
        let input = input.trim();
        if input.is_empty() {
            return;
        }
        let combined = if current_transcript.trim().is_empty() {
            format!("You: {input}")
        } else {
            format!("{current_transcript}\nYou: {input}")
        };
        self.emit(Update::Transcript(combined.clone()));
        self.emit(Update::Status("Assistant thinking...".to_string()));
        let _ = self.llm_tx.send(LlmRequest::Transcript(combined));
    }

    /// Detect issues under discussion and extract Jira change proposals.
    ///
    /// `transcript` is the raw spoken/typed dialogue, used only for cheap
    /// local regex-based issue-key detection (explicit `KEY-123` mentions can
    /// be dropped when the LLM condenses the discussion). `notes` is the
    /// assistant's own running-notes answer for that transcript; it is the
    /// text actually fed into the extraction prompt, so the model re-analyzes
    /// what it has already understood instead of being re-prompted with the
    /// full raw transcript a second time. Falls back to `transcript` when no
    /// notes are available yet (e.g. before the first LLM answer arrives).
    pub fn analyze(&self, transcript: String, notes: String, project_key: String) {
        if transcript.trim().len() < 20 {
            self.emit(Update::Status("Need more transcript before analyzing.".to_string()));
            return;
        }

        self.emit(Update::Analyzing(true));
        self.emit(Update::Status("Resolving issues under discussion...".to_string()));

        let keys = detect_issue_keys(&transcript);
        let detected = if keys.is_empty() {
            "none detected".to_string()
        } else {
            keys.join(", ")
        };
        self.emit(Update::DetectedIssues(detected));

        let notes = if notes.trim().is_empty() { transcript.clone() } else { notes };

        let this = self.clone();
        self.tokio_rt.spawn(async move {
            // Best-effort: fetch context for explicitly-mentioned issues.
            let mut issue_context =
                build_issue_context(&this.jira_client, &this.jira_cfg, &keys, &transcript).await;

            // Fetch previously generated proposals from the generic context
            // store and make the LLM aware of them, so it can prefer
            // referencing/updating one instead of proposing a duplicate.
            let cached: Vec<ProposalItem> = this
                .context
                .list::<ProposalItem>(MODULE_ID, PROPOSALS_COLLECTION)
                .into_iter()
                .map(|(_, item)| item)
                .collect();
            let cached_block = render_cached_for_prompt(&cached);
            if !cached_block.is_empty() {
                if !issue_context.trim().is_empty() {
                    issue_context.push_str("\n\n");
                }
                issue_context.push_str(
                    "Proposals already tracked this session (avoid duplicating; prefer \
                     referencing them if this is the same change):\n",
                );
                issue_context.push_str(&cached_block);
            }

            let default_project = this
                .jira_cfg
                .default_project
                .clone()
                .unwrap_or_else(|| project_key.clone());
            let prompt = build_extraction_prompt(&notes, &issue_context, &default_project);

            // Ask the LLM worker for a structured completion and await its reply.
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<String>();
            if this
                .llm_tx
                .send(LlmRequest::Complete {
                    prompt,
                    max_new_tokens: PROPOSAL_MAX_TOKENS,
                    reply: reply_tx,
                })
                .is_err()
            {
                this.emit(Update::Analyzing(false));
                this.emit(Update::Status("LLM worker unavailable.".to_string()));
                return;
            }
            let raw = match reply_rx.await {
                Ok(raw) => raw,
                Err(_) => {
                    this.emit(Update::Analyzing(false));
                    this.emit(Update::Status("LLM worker stopped before replying.".to_string()));
                    return;
                }
            };

            // Merge each parsed proposal into existing tracked proposals when
            // it looks like the same change (matched via `find_match`),
            // updating it in place instead of appending a duplicate.
            // Otherwise, add it as a new item.
            let parsed = parse_proposals(&raw);
            let mut new_count = 0usize;
            let mut updated_count = 0usize;
            {
                let mut guard = this.items.lock().unwrap();
                for p in parsed {
                    if let Some(idx) = find_match(&p, &guard) {
                        let item = &mut guard[idx];
                        item.proposal = p;
                        item.status = "pending".to_string();
                        item.result = String::new();
                        updated_count += 1;
                    } else {
                        guard.push(ProposalItem {
                            id: this.next_id.fetch_add(1, Ordering::SeqCst),
                            proposal: p,
                            status: "pending".to_string(),
                            result: String::new(),
                        });
                        new_count += 1;
                    }
                }
            }
            this.sync_context();
            this.render_proposals();

            // For any UPDATE whose target key is unknown, search the project for
            // the existing issue and fill it in.
            let matched = resolve_update_targets(
                &this.jira_client,
                &this.jira_cfg,
                &project_key,
                &this.items,
            )
            .await;
            this.render_proposals();

            this.emit(Update::Analyzing(false));
            let extra = if matched > 0 {
                format!(" ({matched} update target(s) matched)")
            } else {
                String::new()
            };
            this.emit(Update::Status(format!(
                "{new_count} new, {updated_count} updated proposal(s) ready for review.{extra}"
            )));
        });
    }

    /// Apply an inline edit to a proposal field (no re-render: the DOM already
    /// holds the user's text, and re-pushing would clobber concurrent edits).
    pub fn update_proposal(&self, id: i32, field: &str, value: String) {
        {
            let mut guard = self.items.lock().unwrap();
            if let Some(item) = guard.iter_mut().find(|i| i.id == id) {
                match field {
                    "summary" => item.proposal.summary = value,
                    "description" => item.proposal.description = value,
                    "acceptance_criteria" => item.proposal.acceptance_criteria = value,
                    "issue_type" => item.proposal.issue_type = value,
                    "target_key" => item.proposal.target_key = value,
                    _ => {}
                }
            }
        }
        self.sync_context();
    }

    /// Reject a proposal (never written to Jira).
    pub fn reject_proposal(&self, id: i32) {
        {
            let mut guard = self.items.lock().unwrap();
            if let Some(item) = guard.iter_mut().find(|i| i.id == id) {
                item.status = "rejected".to_string();
            }
        }
        self.sync_context();
        self.render_proposals();
    }

    /// Approve a (possibly edited) proposal and apply it to Jira.
    pub fn approve_proposal(&self, id: i32, project_key: String) {
        // Snapshot the (possibly edited) proposal and mark it in-flight.
        let proposal = {
            let mut guard = self.items.lock().unwrap();
            match guard.iter_mut().find(|i| i.id == id) {
                Some(item) => {
                    item.status = "approved".to_string();
                    item.result = "Applying...".to_string();
                    item.proposal.clone()
                }
                None => return,
            }
        };
        self.sync_context();
        self.render_proposals();

        let this = self.clone();
        self.tokio_rt.spawn(async move {
            let outcome =
                apply_proposal(&this.jira_client, &this.jira_cfg, &proposal, &project_key).await;
            {
                let mut guard = this.items.lock().unwrap();
                if let Some(item) = guard.iter_mut().find(|i| i.id == id) {
                    match &outcome {
                        Ok(msg) => {
                            item.status = "applied".to_string();
                            item.result = msg.clone();
                        }
                        Err(e) => {
                            item.status = "error".to_string();
                            item.result = format!("Failed: {e}");
                        }
                    }
                }
            }
            this.sync_context();
            this.render_proposals();
        });
    }
}

// --- Proposal caching / merge helpers --------------------------------------

/// Find the index of an existing (non-terminal) [`ProposalItem`] in `items`
/// that represents the same change as `candidate`, so `analyze()` can update
/// it in place instead of appending a duplicate.
///
/// Matching, in priority order:
/// 1. Both have a non-empty `target_key` and they're equal (case-insensitive).
/// 2. Their normalized summaries are exactly equal.
/// 3. Token-overlap similarity of `summary + description` is the highest
///    among candidates and exceeds [`SIMILARITY_MATCH_THRESHOLD`].
///
/// Items already `"applied"` or `"rejected"` are left alone (finalized), so a
/// re-analysis never silently rewrites them; a fresh proposal is created for
/// those instead.
fn find_match(candidate: &Proposal, items: &[ProposalItem]) -> Option<usize> {
    let eligible = || {
        items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.status != "applied" && item.status != "rejected")
    };

    if !candidate.target_key.trim().is_empty() {
        if let Some((idx, _)) = eligible().find(|(_, item)| {
            !item.proposal.target_key.trim().is_empty()
                && item.proposal.target_key.trim().eq_ignore_ascii_case(candidate.target_key.trim())
        }) {
            return Some(idx);
        }
    }

    let candidate_summary = normalized_summary(&candidate.summary);
    if !candidate_summary.is_empty() {
        if let Some((idx, _)) =
            eligible().find(|(_, item)| normalized_summary(&item.proposal.summary) == candidate_summary)
        {
            return Some(idx);
        }
    }

    eligible()
        .map(|(idx, item)| (idx, similarity_score(candidate, &item.proposal)))
        .filter(|(_, score)| *score > SIMILARITY_MATCH_THRESHOLD)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx)
}

/// Render currently-tracked (non-rejected) proposals as a compact bullet list
/// for the LLM extraction prompt, so the model is aware of what's already on
/// the table and can prefer referencing/updating it over proposing a
/// near-duplicate.
fn render_cached_for_prompt(items: &[ProposalItem]) -> String {
    items
        .iter()
        .filter(|item| item.status != "rejected")
        .map(|item| {
            let target = if item.proposal.target_key.trim().is_empty() {
                "(new)".to_string()
            } else {
                item.proposal.target_key.clone()
            };
            format!(
                "- [{}] {} {} :: {}",
                item.proposal.action.as_str(),
                item.proposal.issue_type,
                target,
                item.proposal.summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// --- UI <-> state rendering ------------------------------------------------

fn to_proposal_view(item: &ProposalItem) -> ProposalView {
    ProposalView {
        id: item.id,
        action: item.proposal.action.as_str().to_string(),
        target_key: item.proposal.target_key.clone(),
        issue_type: item.proposal.issue_type.clone(),
        summary: item.proposal.summary.clone(),
        description: item.proposal.description.clone(),
        acceptance_criteria: item.proposal.acceptance_criteria.clone(),
        rationale: item.proposal.rationale.clone(),
        status: item.status.clone(),
        result: item.result.clone(),
    }
}

/// Compose the full Jira description from the proposal's description and
/// acceptance criteria.
fn full_description(p: &Proposal) -> String {
    if p.acceptance_criteria.trim().is_empty() {
        p.description.clone()
    } else {
        format!(
            "{}\n\nAcceptance Criteria:\n{}",
            p.description.trim(),
            p.acceptance_criteria.trim()
        )
    }
}

// --- App wiring ------------------------------------------------------------

fn run_app(llm_ctx: LlmContext) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = ModelConfig::load();
    let jira_cfg = Arc::new(JiraConfig::load());

    let tokio_rt = Arc::new(Runtime::new().expect("Failed to start Tokio runtime"));

    // Shared review state (source of truth for proposals).
    let items: Arc<Mutex<Vec<ProposalItem>>> = Arc::new(Mutex::new(Vec::new()));
    let next_id = Arc::new(AtomicI32::new(1));

    // Lazily-connected Jira MCP client, shared across async tasks.
    let jira_client: Arc<tokio::sync::Mutex<Option<JiraClient>>> =
        Arc::new(tokio::sync::Mutex::new(None));

    // Live transcript (from the recorder) is forwarded into the LLM worker as
    // `LlmRequest::Transcript`; proposal extraction is sent as `LlmRequest::Complete`.
    let (llm_tx, llm_rx) = mpsc::channel::<LlmRequest>();

    // Typed core-event bus (STT + LLM services -> UI) and the module's UI-update
    // channel drained by the Dioxus component.
    let (events, mut core_rx) = crate::core::bus::channel();
    let (update_tx, update_rx) = tokio::sync::mpsc::unbounded_channel::<Update>();

    {
        // Adapt core events into the UI's update stream (single hub, one hop).
        let update_tx = update_tx.clone();
        tokio_rt.spawn(async move {
            while let Some(event) = core_rx.recv().await {
                if update_tx.send(Update::from(event)).is_err() {
                    break;
                }
            }
        });
    }

    let backend = Backend {
        cfg: cfg.clone(),
        jira_cfg,
        tokio_rt,
        jira_client,
        items,
        next_id,
        context: ContextStore::new(),
        llm_tx,
        events: events.clone(),
        update_tx,
        update_rx: Arc::new(Mutex::new(Some(update_rx))),
    };

    llm::spawn_worker(llm_rx, cfg, llm_ctx, events);

    // Custom desktop window configuration.
    let window_config = Config::new().with_window(
        WindowBuilder::new()
            .with_title("Meeting to Jira (Dioxus + Rust)")
            .with_inner_size(dioxus::desktop::LogicalSize::new(720.0, 900.0))
            .with_resizable(true),
    );

    // Launch the Dioxus app, providing the backend handle via context.
    dioxus::LaunchBuilder::desktop()
        .with_cfg(window_config)
        .with_context(backend)
        .launch(AppJira);

    Ok(())
}

// --- Jira interaction helpers ---------------------------------------------

/// Ensure the shared [`JiraClient`] is connected, returning an error string if
/// credentials are missing or the connection fails.
async fn ensure_client(
    client: &Arc<tokio::sync::Mutex<Option<JiraClient>>>,
    cfg: &JiraConfig,
) -> Result<(), String> {
    let mut guard = client.lock().await;
    if guard.is_some() {
        return Ok(());
    }
    if !cfg.is_usable() {
        return Err("Atlassian credentials not configured (set ATLASSIAN_EMAIL / \
ATLASSIAN_API_TOKEN, or JIRA_MCP_MOCK=1)"
            .to_string());
    }
    match JiraClient::connect(cfg).await {
        Ok(c) => {
            *guard = Some(c);
            Ok(())
        }
        Err(e) => Err(format!("{e:#}")),
    }
}

/// Fetch context about the issues under discussion to ground the LLM. Uses
/// explicitly-mentioned keys when present, otherwise a fuzzy keyword JQL search
/// to surface candidate issues. Best effort: returns whatever context we have.
async fn build_issue_context(
    client: &Arc<tokio::sync::Mutex<Option<JiraClient>>>,
    cfg: &JiraConfig,
    keys: &[String],
    transcript: &str,
) -> String {
    if ensure_client(client, cfg).await.is_err() {
        return String::new();
    }

    let guard = client.lock().await;
    let Some(c) = guard.as_ref() else {
        return String::new();
    };

    let mut context = String::new();

    if keys.is_empty() {
        // No explicit key: fuzzy-search candidate issues by keyword.
        if let Some(jql) = keyword_jql(transcript, c.default_project()) {
            if let Ok(candidates) = c.search_jql(&jql, 5).await {
                if !candidates.is_empty() {
                    context.push_str("Candidate issues (verify before updating):\n");
                    for issue in candidates {
                        context.push_str(&format!(
                            "- {} ({}, {}): {}\n",
                            issue.key, issue.issue_type, issue.status, issue.summary
                        ));
                    }
                }
            }
        }
        return context;
    }

    for key in keys {
        if let Ok(issue) = c.get_issue(key).await {
            let fields = issue.get("fields");
            let field = |path: &[&str]| -> String {
                let mut node = fields;
                for p in path {
                    node = node.and_then(|n| n.get(*p));
                }
                node.and_then(|v| v.as_str()).unwrap_or("").to_string()
            };
            context.push_str(&format!(
                "- {key} ({}): {}\n  {}\n",
                field(&["issuetype", "name"]),
                field(&["summary"]),
                field(&["description"]),
            ));
        }
    }
    context
}

/// Build a best-effort fuzzy JQL query from transcript keywords, optionally
/// scoped to a project. Returns `None` when no useful keywords are found.
fn keyword_jql(transcript: &str, project: Option<&str>) -> Option<String> {
    let mut words: Vec<String> = Vec::new();
    for raw in transcript.split(|c: char| !c.is_alphanumeric()) {
        let w = raw.to_lowercase();
        if w.len() >= 5 && raw.chars().all(|c| c.is_alphabetic()) && !words.contains(&w) {
            words.push(w);
        }
        if words.len() >= 5 {
            break;
        }
    }
    if words.is_empty() {
        return None;
    }
    let phrase = words.join(" ").replace('"', "");
    let text_clause = format!("text ~ \"{phrase}\"");
    match project {
        Some(p) if !p.trim().is_empty() => Some(format!("project = {p} AND {text_clause}")),
        _ => Some(text_clause),
    }
}

/// For each UPDATE proposal that has no explicit target key, search the project
/// for the existing issue that best matches the proposal summary and fill in the
/// resolved key (the user still confirms before applying). Returns the number of
/// proposals for which a matching issue was found.
async fn resolve_update_targets(
    client: &Arc<tokio::sync::Mutex<Option<JiraClient>>>,
    cfg: &JiraConfig,
    project_key: &str,
    items: &Arc<Mutex<Vec<ProposalItem>>>,
) -> usize {
    // Snapshot the proposals that need a target resolved.
    let pending: Vec<(i32, String)> = {
        let guard = items.lock().unwrap();
        guard
            .iter()
            .filter(|it| {
                it.proposal.action == ProposalAction::Update
                    && it.proposal.target_key.trim().is_empty()
                    && !it.proposal.summary.trim().is_empty()
            })
            .map(|it| (it.id, it.proposal.summary.clone()))
            .collect()
    };
    if pending.is_empty() {
        return 0;
    }
    if ensure_client(client, cfg).await.is_err() {
        return 0;
    }

    let guard = client.lock().await;
    let Some(c) = guard.as_ref() else {
        return 0;
    };

    let project = if project_key.trim().is_empty() {
        c.default_project().unwrap_or("").to_string()
    } else {
        project_key.to_string()
    };

    let mut resolved = 0usize;
    for (id, summary) in pending {
        let jql = build_update_search_jql(&summary, &project);
        let found = match c.search_jql(&jql, 1).await {
            Ok(v) => v.into_iter().next(),
            Err(_) => None,
        };

        let mut g = items.lock().unwrap();
        if let Some(it) = g.iter_mut().find(|x| x.id == id) {
            match found {
                Some(issue) => {
                    it.proposal.target_key = issue.key.clone();
                    it.result =
                        format!("🔗 Matched existing {} — {}", issue.key, issue.summary);
                    resolved += 1;
                }
                None => {
                    it.result =
                        "No matching issue found in project; set the target key manually."
                            .to_string();
                }
            }
        }
    }
    resolved
}

/// Build a fuzzy JQL query to find an existing issue matching a proposal summary,
/// scoped to `project` when provided. Prefers the most recently updated match.
fn build_update_search_jql(summary: &str, project: &str) -> String {
    let words: Vec<String> = summary
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
        .take(6)
        .map(|w| w.to_lowercase())
        .collect();
    let phrase = if words.is_empty() {
        summary.trim().replace('"', " ")
    } else {
        words.join(" ")
    };
    let text_clause = format!("text ~ \"{phrase}\"");
    if project.trim().is_empty() {
        format!("{text_clause} ORDER BY updated DESC")
    } else {
        format!("project = {project} AND {text_clause} ORDER BY updated DESC")
    }
}

/// Apply a single approved proposal to Jira, returning a human-readable result.
async fn apply_proposal(
    client: &Arc<tokio::sync::Mutex<Option<JiraClient>>>,
    cfg: &JiraConfig,
    proposal: &Proposal,
    project_key: &str,
) -> Result<String, String> {
    ensure_client(client, cfg).await?;
    let guard = client.lock().await;
    let c = guard.as_ref().ok_or("Jira client unavailable")?;

    let description = full_description(proposal);

    match proposal.action {
        ProposalAction::Create => {
            let project = if project_key.trim().is_empty() {
                c.default_project().unwrap_or("PROJ")
            } else {
                project_key
            };
            let result = c
                .create_issue(project, &proposal.issue_type, &proposal.summary, &description)
                .await
                .map_err(|e| format!("{e:#}"))?;
            let key = created_key(&result).unwrap_or_else(|| "(unknown key)".to_string());
            Ok(format!("✅ Created {key}"))
        }
        ProposalAction::Update => {
            let key = proposal.target_key.trim();
            if key.is_empty() {
                return Err("Update proposal has no target issue key".to_string());
            }
            c.edit_issue(key, &proposal.summary, &proposal.description)
                .await
                .map_err(|e| format!("{e:#}"))?;

            // Record acceptance criteria as a comment so reviewers see the
            // rationale in Jira's history rather than silently in the body.
            if !proposal.acceptance_criteria.trim().is_empty() {
                let comment = format!(
                    "Acceptance criteria (from meeting):\n{}",
                    proposal.acceptance_criteria.trim()
                );
                c.add_comment(key, &comment)
                    .await
                    .map_err(|e| format!("{e:#}"))?;
            }
            Ok(format!("✅ Updated {key}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: i32, action: ProposalAction, target_key: &str, summary: &str, status: &str) -> ProposalItem {
        ProposalItem {
            id,
            proposal: Proposal {
                action,
                target_key: target_key.to_string(),
                issue_type: "Story".to_string(),
                summary: summary.to_string(),
                description: "Users can reset their password via an emailed link.".to_string(),
                acceptance_criteria: String::new(),
                rationale: String::new(),
            },
            status: status.to_string(),
            result: String::new(),
        }
    }

    fn candidate(action: ProposalAction, target_key: &str, summary: &str, description: &str) -> Proposal {
        Proposal {
            action,
            target_key: target_key.to_string(),
            issue_type: "Story".to_string(),
            summary: summary.to_string(),
            description: description.to_string(),
            acceptance_criteria: String::new(),
            rationale: String::new(),
        }
    }

    #[test]
    fn find_match_by_exact_target_key() {
        let items = vec![item(1, ProposalAction::Update, "PROJ-42", "Trim onboarding epic", "pending")];
        let c = candidate(ProposalAction::Update, "PROJ-42", "Different wording entirely", "unrelated text");
        assert_eq!(find_match(&c, &items), Some(0));
    }

    #[test]
    fn find_match_by_exact_normalized_summary() {
        let items = vec![item(1, ProposalAction::Create, "", "Add password reset", "pending")];
        let c = candidate(ProposalAction::Create, "", "  ADD   password reset  ", "some other description");
        assert_eq!(find_match(&c, &items), Some(0));
    }

    #[test]
    fn find_match_by_similarity_fallback() {
        let items = vec![item(
            1,
            ProposalAction::Create,
            "",
            "Add password reset via email",
            "pending",
        )];
        let c = candidate(
            ProposalAction::Create,
            "",
            "Add password reset by email",
            "Users can reset their password via an emailed link.",
        );
        assert_eq!(find_match(&c, &items), Some(0));
    }

    #[test]
    fn find_match_returns_none_for_unrelated_proposal() {
        let items = vec![item(1, ProposalAction::Create, "", "Add password reset", "pending")];
        let c = candidate(
            ProposalAction::Create,
            "",
            "Refactor CI pipeline caching",
            "Speed up docker layer builds by caching more aggressively.",
        );
        assert_eq!(find_match(&c, &items), None);
    }

    #[test]
    fn find_match_ignores_finalized_items() {
        let items = vec![
            item(1, ProposalAction::Create, "", "Add password reset", "applied"),
            item(2, ProposalAction::Create, "", "Add password reset", "rejected"),
        ];
        let c = candidate(ProposalAction::Create, "", "Add password reset", "same idea again");
        assert_eq!(find_match(&c, &items), None);
    }

    #[test]
    fn render_cached_for_prompt_skips_rejected_and_formats_pending() {
        let items = vec![
            item(1, ProposalAction::Create, "", "Add password reset", "pending"),
            item(2, ProposalAction::Update, "PROJ-42", "Trim onboarding epic", "rejected"),
        ];
        let rendered = render_cached_for_prompt(&items);
        assert!(rendered.contains("Add password reset"));
        assert!(!rendered.contains("Trim onboarding epic"));
    }
}
