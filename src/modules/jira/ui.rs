//! Dioxus screen: the "Meeting to Jira Story
//! Assistant". Owns the reactive UI state (signals) and renders the layout using
//! the local `vtmn-play` (Vitamin Play) CSS. All backend behaviour lives in the
//! shared [`Backend`](super::Backend); this module only reflects state and
//! forwards user intent.

use dioxus::prelude::*;

use super::Backend;
use crate::core::CoreEvent;

static MAIN_CSS: Asset = asset!("assets/vtmn-play/css/index.css");

/// A single reviewable proposal, in a form the UI can render/diff.
#[derive(Clone, PartialEq)]
pub struct ProposalView {
    pub id: i32,
    /// "create" | "update".
    pub action: String,
    pub target_key: String,
    pub issue_type: String,
    pub summary: String,
    pub description: String,
    pub acceptance_criteria: String,
    pub rationale: String,
    pub priority: String,
    /// Comma-separated labels for display/editing.
    pub labels: String,
    pub assignee: String,
    pub sprint: String,
    /// Bulk-selection checkbox state.
    pub selected: bool,
    /// "pending" | "approved" | "rejected" | "applied" | "error".
    pub status: String,
    pub result: String,
    /// Current field values fetched from Jira for `update` proposals, used to
    /// render an old→new diff. `None` when not yet fetched (or a create).
    pub current_summary: Option<String>,
    pub current_description: Option<String>,
    pub current_priority: Option<String>,
    pub current_labels: Option<String>,
    pub current_assignee: Option<String>,
    pub current_sprint: Option<String>,
}

/// Messages pushed from backend threads/tasks into the UI. Consumed by a task
/// running on the Dioxus runtime, which writes the corresponding signal.
pub enum Update {
    Status(String),
    Transcript(String),
    Response(String),
    DetectedIssues(String),
    Analyzing(bool),
    Proposals(Vec<ProposalView>),
}

impl From<CoreEvent> for Update {
    fn from(event: CoreEvent) -> Self {
        match event {
            CoreEvent::Status(s) => Update::Status(s),
            CoreEvent::Transcript(s) => Update::Transcript(s),
            CoreEvent::Response(s) => Update::Response(s),
        }
    }
}

/// Root component: reproduces the Slint layout and wires signals to [`Backend`].
#[component]
pub fn AppJira() -> Element {
    let backend = use_context::<Backend>();

    // --- Reactive state (mirrors the old Slint in/in-out properties) ---
    let is_recording = use_signal(|| false);
    let analyzing = use_signal(|| false);
    let status_msg = use_signal(|| "Idle - Ready to record".to_string());
    let transcript = use_signal(String::new);
    let llm_response = use_signal(String::new);
    let detected_issues = use_signal(String::new);
    let project_key = use_signal(|| backend.jira_cfg.default_project.clone().unwrap_or_default());
    let mut input_text = use_signal(String::new);
    let proposals = use_signal(Vec::<ProposalView>::new);
    let mut last_analyzed_len = use_signal(|| 0usize);

    // Expose the project key to proposal cards (used when applying).
    use_context_provider(|| ProjectKeySignal(project_key));

    // Non-Send microphone recorder: kept on the main thread (cpal::Stream is
    // !Send), started/stopped inside the toggle handler.
    let recorder = use_hook(|| {
        std::rc::Rc::new(std::cell::RefCell::new(
            None::<crate::core::stt::Recorder>,
        ))
    });

    // Drain backend updates into the signals. `use_hook` runs once; `spawn`
    // schedules the task on the Dioxus runtime so signal writes are valid.
    use_hook(|| {
        let backend_cl = backend.clone();
        let mut analyzing = analyzing;
        let mut status_msg = status_msg;
        let mut transcript = transcript;
        let mut llm_response = llm_response;
        let mut detected_issues = detected_issues;
        let mut proposals = proposals;
        let mut rx = backend_cl.take_update_rx();
        spawn(async move {
            if let Some(rx) = rx.as_mut() {
                while let Some(update) = rx.recv().await {
                    match update {
                        Update::Status(s) => status_msg.set(s),
                        Update::Transcript(s) => transcript.set(s),
                        Update::Response(s) => llm_response.set(s),
                        Update::DetectedIssues(s) => detected_issues.set(s),
                        Update::Analyzing(b) => analyzing.set(b),
                        Update::Proposals(p) => proposals.set(p),
                    }
                }
            }
        });
    });

    // --- Event handlers ---
    let on_toggle = {
        let backend = backend.clone();
        let recorder = recorder.clone();
        let mut is_recording = is_recording;
        let mut status_msg = status_msg;
        let mut transcript = transcript;
        let mut llm_response = llm_response;
        move |_| {
            if !is_recording() {
                transcript.set(String::new());
                llm_response.set(String::new());
                last_analyzed_len.set(0);
                match backend.start_recording() {
                    Ok(rec) => {
                        *recorder.borrow_mut() = Some(rec);
                        is_recording.set(true);
                        status_msg.set("Recording & transcribing live...".to_string());
                    }
                    Err(e) => {
                        is_recording.set(false);
                        status_msg.set(format!("Cannot start recording: {e}"));
                    }
                }
            } else {
                if let Some(rec) = recorder.borrow_mut().take() {
                    rec.stop();
                }
                is_recording.set(false);
                status_msg
                    .set("Stopped. Proposals are extracted automatically as you speak.".to_string());
            }
        }
    };

    let send_message = {
        let backend = backend.clone();
        let mut input_text = input_text;
        let transcript = transcript;
        move || {
            let input = input_text().trim().to_string();
            if input.is_empty() {
                return;
            }
            backend.send_message(transcript(), input);
            input_text.set(String::new());
        }
    };

    let on_analyze = {
        let backend = backend.clone();
        move |transcript_text: String, notes_text: String, project: String| {
            backend.analyze(transcript_text, notes_text, project)
        }
    };

    // Auto-analyze: once the assistant's running notes (the LLM's own answer,
    // not the raw transcript) grow by the configured number of characters
    // since the last analysis, extract Jira proposals automatically — no
    // button press required. Keying off the notes instead of the raw
    // transcript means extraction re-analyzes what the model has already
    // condensed rather than re-prompting it with the full raw dialogue again.
    // The `analyzing` flag prevents overlapping runs; text that arrives
    // mid-analysis is picked up on the next notes change once analysis
    // completes.
    use_effect({
        let backend = backend.clone();
        let on_analyze = on_analyze.clone();
        let transcript = transcript;
        let llm_response = llm_response;
        let analyzing = analyzing;
        let project_key = project_key;
        let mut last_analyzed_len = last_analyzed_len;
        move || {
            let notes = llm_response();
            let current_transcript = transcript();
            let len = notes.len();
            let threshold = backend.analyze_threshold();
            if !analyzing()
                && current_transcript.trim().len() >= 20
                && len.saturating_sub(last_analyzed_len()) >= threshold
            {
                last_analyzed_len.set(len);
                on_analyze(current_transcript, notes, project_key());
            }
        }
    });

    let recording = is_recording();
    let analyzing_now = analyzing();
    let status_color = if recording { "#d9534f" } else { "#5bc0de" };
    let transcript_val = transcript();
    let response_val = llm_response();
    let detected_val = detected_issues();
    let proposals_list = proposals();

    rsx! {
        document::Stylesheet { href: MAIN_CSS }
        main {
            style: "max-width: 620px; margin: 0 auto; padding: 16px; font-family: system-ui, sans-serif; display: flex; flex-direction: column; gap: 10px;",

            // --- Title ---
            h1 {
                style: "font-size: 18px; font-weight: 700; margin: 0;",
                "🎙️ Meeting to Jira Story Assistant"
            }

            // --- Controls ---
            div {
                style: "display: flex; gap: 8px;",
                button {
                    class: "vp-button vp-button--small",
                    onclick: on_toggle,
                    if recording { "⏹️ Stop" } else { "🔴 Start Recording" }
                }
                if analyzing_now {
                    span {
                        style: "align-self: center; color: #5bc0de;",
                        "🔎 Analyzing…"
                    }
                }
            }

            // --- Status ---
            div {
                style: "color: {status_color};",
                "Status: {status_msg}"
            }

            // --- Text input (alternative to speech) ---
            div {
                style: "display: flex; gap: 8px;",
                input {
                    class: "vp-input",
                    style: "flex: 1;",
                    placeholder: "Type a message to the assistant, then press Enter…",
                    value: "{input_text}",
                    oninput: move |evt| input_text.set(evt.value()),
                    onkeydown: {
                        let mut send_message = send_message.clone();
                        move |evt: KeyboardEvent| {
                            if evt.key() == Key::Enter {
                                send_message();
                            }
                        }
                    }
                }
                button {
                    class: "vp-button vp-button--small",
                    disabled: input_text().is_empty(),
                    onclick: {
                        let mut send_message = send_message.clone();
                        move |_| send_message()
                    },
                    "➤ Send"
                }
            }

            // --- Live transcript ---
            div { style: "font-size: 14px; font-weight: 600;", "Live Transcript" }
            div {
                style: "height: 90px; overflow-y: auto; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px; color: #222; white-space: pre-wrap;",
                if transcript_val.is_empty() {
                    "(transcript will appear here while recording…)"
                } else {
                    "{transcript_val}"
                }
            }

            // --- Assistant response ---
            div { style: "font-size: 14px; font-weight: 600;", "Assistant Notes" }
            div {
                style: "height: 70px; overflow-y: auto; background: #f4f8ff; border: 1px solid #cfe0ff; border-radius: 4px; padding: 6px; color: #143; white-space: pre-wrap;",
                if response_val.is_empty() {
                    "(the assistant's running notes appear here…)"
                } else {
                    "{response_val}"
                }
            }

            div { style: "height: 1px; background: #ccc;" }

            // --- Configuration + detected issues ---
            div {
                style: "display: flex; align-items: center; gap: 8px;",
                span { "Jira Project Key:" }
                input {
                    class: "vp-input vp-input--small",
                    style: "flex: 1;",
                    value: "{project_key}",
                    oninput: {
                        let mut project_key = project_key;
                        move |evt| project_key.set(evt.value())
                    }
                }
            }
            if !detected_val.is_empty() {
                div {
                    style: "color: #555;",
                    "Issues under discussion: {detected_val}"
                }
            }

            div {
                style: "font-size: 14px; font-weight: 600;",
                "Proposed Changes (review & edit before applying)"
            }

            if proposals_list.is_empty() {
                div {
                    style: "color: #888;",
                    "No proposals yet. They appear automatically as the discussion grows."
                }
            } else {
                div {
                    style: "display: flex; gap: 8px;",
                    button {
                        class: "vp-button vp-button--small",
                        onclick: {
                            let backend = backend.clone();
                            let project_key = project_key;
                            move |_| backend.approve_selected(project_key.peek().clone())
                        },
                        "✅ Approve selected"
                    }
                    button {
                        class: "vp-button vp-button--small vp-button--negative",
                        onclick: {
                            let backend = backend.clone();
                            move |_| backend.reject_selected()
                        },
                        "🗑️ Reject selected"
                    }
                }
            }

            div {
                style: "display: flex; flex-direction: column; gap: 8px;",
                for p in proposals_list.iter().cloned() {
                    ProposalCard { key: "{p.id}", proposal: p }
                }
            }
        }
    }
}

/// One reviewable proposal card with inline editing and approve/reject actions.
#[component]
fn ProposalCard(proposal: ProposalView) -> Element {
    let backend = use_context::<Backend>();
    let project_key_ctx = use_context::<ProjectKeySignal>().0;

    let id = proposal.id;
    let is_update = proposal.action == "update";
    let bg = match proposal.status.as_str() {
        "applied" => "#eaf7ea",
        "rejected" => "#f2f2f2",
        "error" => "#fdecea",
        _ => "#f8f9fa",
    };
    let action_label = if is_update { "✏️ UPDATE" } else { "➕ CREATE" };
    let action_color = if is_update { "#b8860b" } else { "#2e7d32" };
    let actionable = proposal.status == "pending" || proposal.status == "approved";

    let update_field = {
        let backend = backend.clone();
        move |field: &'static str, value: String| {
            backend.update_proposal(id, field, value);
        }
    };

    rsx! {
        div {
            style: "background: {bg}; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px; display: flex; flex-direction: column; gap: 4px;",

            // Header row: checkbox + action + type + target
            div {
                style: "display: flex; align-items: center; gap: 8px; flex-wrap: wrap;",
                input {
                    r#type: "checkbox",
                    checked: proposal.selected,
                    onchange: {
                        let backend = backend.clone();
                        move |evt: FormEvent| backend.toggle_selected(id, evt.checked())
                    }
                }
                span {
                    style: "font-weight: 700; color: {action_color};",
                    "{action_label}"
                }
                span { "Type:" }
                input {
                    class: "vp-input vp-input--small",
                    style: "width: 90px;",
                    value: "{proposal.issue_type}",
                    oninput: {
                        let update_field = update_field.clone();
                        move |evt| update_field("issue_type", evt.value())
                    }
                }
                span { "Target:" }
                input {
                    class: "vp-input vp-input--small",
                    style: "width: 100px;",
                    placeholder: "KEY-123",
                    value: "{proposal.target_key}",
                    oninput: {
                        let update_field = update_field.clone();
                        move |evt| update_field("target_key", evt.value())
                    }
                }
            }

            if is_update && proposal.current_summary.is_some() {
                DiffBlock { proposal: proposal.clone() }
            }

            span { style: "font-size: 12px; color: #666;", "Summary" }
            input {
                class: "vp-input",
                value: "{proposal.summary}",
                oninput: {
                    let update_field = update_field.clone();
                    move |evt| update_field("summary", evt.value())
                }
            }

            span { style: "font-size: 12px; color: #666;", "Description" }
            textarea {
                class: "vp-textarea",
                style: "height: 60px; resize: vertical;",
                value: "{proposal.description}",
                oninput: {
                    let update_field = update_field.clone();
                    move |evt| update_field("description", evt.value())
                }
            }

            span { style: "font-size: 12px; color: #666;", "Acceptance criteria" }
            textarea {
                class: "vp-textarea",
                style: "height: 48px; resize: vertical;",
                value: "{proposal.acceptance_criteria}",
                oninput: {
                    let update_field = update_field.clone();
                    move |evt| update_field("acceptance_criteria", evt.value())
                }
            }

            div {
                style: "display: flex; gap: 8px; flex-wrap: wrap;",
                div {
                    style: "flex: 1; min-width: 100px;",
                    span { style: "font-size: 12px; color: #666;", "Priority" }
                    input {
                        class: "vp-input vp-input--small",
                        value: "{proposal.priority}",
                        oninput: {
                            let update_field = update_field.clone();
                            move |evt| update_field("priority", evt.value())
                        }
                    }
                }
                div {
                    style: "flex: 1; min-width: 100px;",
                    span { style: "font-size: 12px; color: #666;", "Assignee" }
                    input {
                        class: "vp-input vp-input--small",
                        value: "{proposal.assignee}",
                        oninput: {
                            let update_field = update_field.clone();
                            move |evt| update_field("assignee", evt.value())
                        }
                    }
                }
                div {
                    style: "flex: 1; min-width: 100px;",
                    span { style: "font-size: 12px; color: #666;", "Sprint" }
                    input {
                        class: "vp-input vp-input--small",
                        value: "{proposal.sprint}",
                        oninput: {
                            let update_field = update_field.clone();
                            move |evt| update_field("sprint", evt.value())
                        }
                    }
                }
            }

            span { style: "font-size: 12px; color: #666;", "Labels (comma-separated)" }
            input {
                class: "vp-input",
                value: "{proposal.labels}",
                oninput: {
                    let update_field = update_field.clone();
                    move |evt| update_field("labels", evt.value())
                }
            }

            if !proposal.rationale.is_empty() {
                div {
                    style: "font-size: 11px; color: #777;",
                    "Why: {proposal.rationale}"
                }
            }

            if (proposal.status == "applied" || proposal.status == "error") && !proposal.result.is_empty() {
                div {
                    style: if proposal.status == "error" { "color: #c0392b;" } else { "color: #2e7d32;" },
                    "{proposal.result}"
                }
            }

            if actionable && !proposal.result.is_empty() {
                div {
                    style: "color: #1565c0; font-style: italic;",
                    "{proposal.result}"
                }
            }

            if actionable {
                div {
                    style: "display: flex; justify-content: flex-end; gap: 8px;",
                    button {
                        class: "vp-button vp-button--small",
                        disabled: proposal.status != "pending",
                        onclick: {
                            let backend = backend.clone();
                            let project_key_ctx = project_key_ctx;
                            move |_| backend.approve_proposal(id, project_key_ctx.peek().clone())
                        },
                        "✅ Approve & Apply"
                    }
                    button {
                        class: "vp-button vp-button--small vp-button--negative",
                        onclick: {
                            let backend = backend.clone();
                            move |_| backend.reject_proposal(id)
                        },
                        "🗑️ Reject"
                    }
                }
            }
        }
    }
}

/// Renders an old→new diff for each changed field of an `update` proposal,
/// using the snapshot fetched from Jira before the change is applied. Only
/// fields that actually differ are shown; unchanged fields are omitted to
/// keep the card compact.
#[component]
fn DiffBlock(proposal: ProposalView) -> Element {
    let rows: Vec<(&'static str, String, String)> = [
        ("Summary", proposal.current_summary.clone(), proposal.summary.clone()),
        ("Description", proposal.current_description.clone(), proposal.description.clone()),
        ("Priority", proposal.current_priority.clone(), proposal.priority.clone()),
        ("Labels", proposal.current_labels.clone(), proposal.labels.clone()),
        ("Assignee", proposal.current_assignee.clone(), proposal.assignee.clone()),
        ("Sprint", proposal.current_sprint.clone(), proposal.sprint.clone()),
    ]
    .into_iter()
    .filter_map(|(label, old, new)| {
        let old = old.unwrap_or_default();
        if old.trim() == new.trim() {
            None
        } else {
            Some((label, old, new))
        }
    })
    .collect();

    if rows.is_empty() {
        return rsx! {};
    }

    rsx! {
        div {
            style: "background: #fffbea; border: 1px solid #f0e0a0; border-radius: 4px; padding: 6px; font-size: 12px; display: flex; flex-direction: column; gap: 4px;",
            span { style: "font-weight: 600; color: #7a5c00;", "Changes vs. current Jira value" }
            for (label, old, new) in rows {
                div {
                    style: "display: flex; flex-direction: column;",
                    span { style: "color: #999; font-weight: 600;", "{label}" }
                    span {
                        style: "color: #b71c1c; text-decoration: line-through;",
                        if old.is_empty() { "(empty)" } else { "{old}" }
                    }
                    span {
                        style: "color: #2e7d32;",
                        if new.is_empty() { "(empty)" } else { "{new}" }
                    }
                }
            }
        }
    }
}

/// Wrapper so the current project key is reachable from proposal cards.
#[derive(Clone, Copy)]
pub struct ProjectKeySignal(pub Signal<String>);
