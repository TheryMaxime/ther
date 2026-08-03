//! Jira change proposals: the structured, reviewable output the assistant
//! derives from the meeting transcript.
//!
//! A [`Proposal`] is either a *create* (new story/epic/task) or an *update* to
//! an existing issue. Proposals are produced by prompting the embedded LLM for
//! strict JSON, parsed defensively here, then surfaced in the UI for the user to
//! edit and approve. Nothing is written to Jira without explicit approval.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Whether a proposal creates a new issue or updates an existing one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProposalAction {
    Create,
    Update,
}

impl ProposalAction {
    fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "update" | "edit" | "modify" => ProposalAction::Update,
            _ => ProposalAction::Create,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ProposalAction::Create => "create",
            ProposalAction::Update => "update",
        }
    }
}

/// A single reviewable Jira change.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub action: ProposalAction,
    /// Target issue key for updates (e.g. `PROJ-42`); empty for creates.
    pub target_key: String,
    /// Jira issue type for creates (Story, Epic, Task, Bug).
    pub issue_type: String,
    pub summary: String,
    pub description: String,
    pub acceptance_criteria: String,
    /// Why the assistant proposed this (shown to the reviewer, not sent to Jira).
    pub rationale: String,
}

/// Matches explicit Jira issue keys such as `PROJ-42` or `AB12-7`.
static ISSUE_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][A-Z0-9]+-\d+\b").unwrap());

/// Detect explicit Jira issue keys mentioned in the transcript, de-duplicated
/// while preserving first-seen order.
pub fn detect_issue_keys(transcript: &str) -> Vec<String> {
    let mut seen = Vec::new();
    for m in ISSUE_KEY_RE.find_iter(transcript) {
        let key = m.as_str().to_string();
        if !seen.contains(&key) {
            seen.push(key);
        }
    }
    seen
}

/// Normalize a summary for exact-match comparison: trim, lowercase, collapse
/// runs of whitespace to a single space.
pub fn normalized_summary(s: &str) -> String {
    s.trim().to_ascii_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Lowercased, deduplicated word set of `summary + description`, used for the
/// token-overlap similarity fallback below.
fn token_set(p: &Proposal) -> HashSet<String> {
    format!("{} {}", p.summary, p.description)
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Jaccard-style token-overlap similarity between two proposals' text
/// (`summary` + `description`), in `[0.0, 1.0]`. Used as a fallback match
/// signal when neither `target_key` nor the normalized summary match exactly.
pub fn similarity_score(a: &Proposal, b: &Proposal) -> f32 {
    let sa = token_set(a);
    let sb = token_set(b);
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let intersection = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Build the strict-JSON extraction prompt for the LLM.
///
/// `notes` is the assistant's own running-notes answer for the discussion so
/// far (not the raw transcript): extraction re-analyzes what the model has
/// already understood/condensed instead of re-prompting it with the full raw
/// dialogue. `issue_context` carries any known detail about issues currently
/// under discussion (fetched from Jira) so updates target the right issue.
pub fn build_extraction_prompt(notes: &str, issue_context: &str, default_project: &str) -> String {
    let context_block = if issue_context.trim().is_empty() {
        "(no existing issues resolved yet)".to_string()
    } else {
        issue_context.trim().to_string()
    };
    let project_hint = if default_project.trim().is_empty() {
        "the current project".to_string()
    } else {
        default_project.trim().to_string()
    };

    format!(
        "[INST] You extract concrete Jira user-story / epic changes agreed in an \
engineering meeting. Default project is {project_hint}.\n\n\
Known existing issues under discussion:\n{context_block}\n\n\
Rules:\n\
- Output ONLY a single JSON object. No prose, no explanation, no code fences.\n\
- \"action\" is exactly \"create\" or \"update\" (never \"delete\").\n\
- Use \"update\" only when an existing KEY-123 is clearly the subject; else \"create\".\n\
- \"target_key\" is set only for updates, else \"\".\n\
- If nothing is actionable, output {{\"proposals\":[]}}.\n\n\
Example output:\n\
{{\"proposals\":[{{\"action\":\"create\",\"target_key\":\"\",\"issue_type\":\"Story\",\
\"summary\":\"Add password reset via email\",\"description\":\"Users request a reset \
link by email that expires after one hour.\",\"acceptance_criteria\":\"Link expires in 1h; \
one active link per user.\",\"rationale\":\"Team agreed users need self-service reset.\"}}]}}\n\n\
Assistant's running notes so far:\n{notes}\n\n\
Now output the JSON object for these notes: [/INST]\n{PROPOSAL_PRIME}"
    )
}

/// The JSON prefix we append to the prompt to *prime* the model into emitting
/// the proposal list directly (small models follow format far better when the
/// answer is already started for them).
pub const PROPOSAL_PRIME: &str = "{\"proposals\": [";

/// Defensively parse the LLM's raw output into proposals.
///
/// Tolerates code fences and surrounding prose by scanning every balanced
/// `{ .. }` / `[ .. ]` span in the text and using the first that parses into a
/// non-empty proposal list. Accepts either a `{"proposals":[..]}` wrapper or a
/// bare array.
pub fn parse_proposals(raw: &str) -> Vec<Proposal> {
    // 1) Prefer a wrapper object / array that yields a full list.
    for span in json_spans(raw) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(span) else {
            continue;
        };
        let array = value
            .get("proposals")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| value.as_array().cloned());
        if let Some(array) = array {
            let proposals: Vec<Proposal> = array
                .iter()
                .filter_map(proposal_from_value)
                .filter(|p| !p.summary.trim().is_empty())
                .collect();
            if !proposals.is_empty() {
                return proposals;
            }
        }
    }

    // 2) Fallback (e.g. primed output with no outer wrapper): collect every
    // individual proposal-shaped object we can find.
    let mut proposals = Vec::new();
    for span in json_spans(raw) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(span) else {
            continue;
        };
        let looks_like_proposal =
            value.is_object() && (value.get("summary").is_some() || value.get("action").is_some());
        if looks_like_proposal {
            if let Some(p) = proposal_from_value(&value) {
                if !p.summary.trim().is_empty() {
                    proposals.push(p);
                }
            }
        }
    }
    proposals
}

fn proposal_from_value(value: &serde_json::Value) -> Option<Proposal> {
    let get = |k: &str| value.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let description = get("description");
    let mut summary = get("summary");
    // Small models often omit `summary`; derive a title from the description.
    if summary.is_empty() && !description.is_empty() {
        summary = derive_summary(&description);
    }
    Some(Proposal {
        action: ProposalAction::from_str(&get("action")),
        target_key: get("target_key"),
        issue_type: {
            let t = get("issue_type");
            if t.is_empty() { "Story".to_string() } else { t }
        },
        summary,
        description,
        acceptance_criteria: get("acceptance_criteria"),
        rationale: get("rationale"),
    })
}

/// Derive a short title from a description (first sentence, capped length).
fn derive_summary(description: &str) -> String {
    let first = description
        .split(|c| c == '.' || c == '\n')
        .next()
        .unwrap_or(description)
        .trim();
    let base = if first.is_empty() { description } else { first };
    let mut s: String = base.chars().take(80).collect();
    if base.chars().count() > 80 {
        s.push('…');
    }
    s
}

/// Yield every top-level balanced `{..}` / `[..]` span in `raw`, in order.
fn json_spans(raw: &str) -> Vec<&str> {
    let bytes = raw.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let (open, close) = match bytes[i] {
            b'{' => (b'{', b'}'),
            b'[' => (b'[', b']'),
            _ => {
                i += 1;
                continue;
            }
        };
        let mut depth = 0i32;
        let mut j = i;
        while j < bytes.len() {
            if bytes[j] == open {
                depth += 1;
            } else if bytes[j] == close {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            j += 1;
        }
        if depth == 0 && j < bytes.len() {
            spans.push(&raw[i..=j]);
            i = j + 1;
        } else {
            // Unbalanced open (e.g. a primed, unterminated wrapper): skip just
            // this char so nested balanced objects remain discoverable.
            i += 1;
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_and_dedups_issue_keys() {
        let t = "We should update PROJ-42 and also PROJ-42, plus AB12-7. Not a-1 or lowercase.";
        assert_eq!(detect_issue_keys(t), vec!["PROJ-42", "AB12-7"]);
    }

    #[test]
    fn parses_proposals_with_code_fence_and_prose() {
        let raw = "Sure! Here is the JSON:\n```json\n{\"proposals\":[\
{\"action\":\"create\",\"issue_type\":\"Story\",\"summary\":\"Add password reset\",\
\"description\":\"Users can reset via email.\",\"acceptance_criteria\":\"link expires\",\
\"rationale\":\"team agreed\"},\
{\"action\":\"update\",\"target_key\":\"PROJ-42\",\"summary\":\"Refine epic\",\
\"description\":\"scope trimmed\"}]}\n```\nHope that helps!";
        let props = parse_proposals(raw);
        assert_eq!(props.len(), 2);
        assert_eq!(props[0].action, ProposalAction::Create);
        assert_eq!(props[0].issue_type, "Story");
        assert_eq!(props[0].summary, "Add password reset");
        assert_eq!(props[1].action, ProposalAction::Update);
        assert_eq!(props[1].target_key, "PROJ-42");
        // Missing issue_type defaults to Story.
        assert_eq!(props[1].issue_type, "Story");
    }

    #[test]
    fn parses_bare_array_and_skips_empty_summary() {
        let raw = "[{\"action\":\"create\",\"summary\":\"\"},\
{\"action\":\"create\",\"summary\":\"Real one\"}]";
        let props = parse_proposals(raw);
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].summary, "Real one");
    }

    #[test]
    fn empty_on_garbage() {
        assert!(parse_proposals("no json here").is_empty());
        assert!(parse_proposals("{\"proposals\":[]}").is_empty());
    }

    #[test]
    fn normalized_summary_collapses_case_and_whitespace() {
        assert_eq!(
            normalized_summary("  Add   Password   RESET \n"),
            "add password reset"
        );
    }

    fn proposal_with(summary: &str, description: &str) -> Proposal {
        Proposal {
            action: ProposalAction::Create,
            target_key: String::new(),
            issue_type: "Story".to_string(),
            summary: summary.to_string(),
            description: description.to_string(),
            acceptance_criteria: String::new(),
            rationale: String::new(),
        }
    }

    #[test]
    fn similarity_score_is_one_for_identical_text() {
        let a = proposal_with("Add password reset", "Users reset via email link");
        let b = proposal_with("Add password reset", "Users reset via email link");
        assert_eq!(similarity_score(&a, &b), 1.0);
    }

    #[test]
    fn similarity_score_is_high_for_similar_text() {
        let a = proposal_with("Add password reset via email", "Link expires after one hour");
        let b = proposal_with("Add password reset by email", "Link expires in one hour");
        let score = similarity_score(&a, &b);
        assert!(score > 0.6, "expected high similarity, got {score}");
    }

    #[test]
    fn similarity_score_is_low_for_unrelated_text() {
        let a = proposal_with("Add password reset via email", "Link expires after one hour");
        let b = proposal_with("Refactor CI pipeline caching", "Speed up docker layer builds");
        let score = similarity_score(&a, &b);
        assert!(score < 0.3, "expected low similarity, got {score}");
    }

    #[test]
    fn proposal_action_serde_roundtrip_matches_str_forms() {
        let create = serde_json::to_string(&ProposalAction::Create).unwrap();
        let update = serde_json::to_string(&ProposalAction::Update).unwrap();
        assert_eq!(create, "\"create\"");
        assert_eq!(update, "\"update\"");
        assert_eq!(
            serde_json::from_str::<ProposalAction>("\"update\"").unwrap(),
            ProposalAction::Update
        );
    }

    #[test]
    fn parses_primed_individual_objects() {
        // Simulates PRIMED output: the model continues after `{"proposals": [`
        // without a proper wrapper/closing.
        let raw = "\n  {\"action\":\"create\",\"issue_type\":\"Story\",\"summary\":\"Add reset\",\
\"description\":\"via email\"},\n  {\"action\":\"update\",\"target_key\":\"PROJ-42\",\
\"summary\":\"Trim epic\",\"description\":\"drop sms\"}\n";
        let props = parse_proposals(raw);
        assert_eq!(props.len(), 2);
        assert_eq!(props[0].summary, "Add reset");
        assert_eq!(props[1].target_key, "PROJ-42");
    }
}
