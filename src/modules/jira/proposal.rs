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
    /// Jira priority name (e.g. "High"); empty when not specified.
    #[serde(default)]
    pub priority: String,
    /// Jira labels; empty when not specified.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Assignee display name or account id; empty when not specified.
    #[serde(default)]
    pub assignee: String,
    /// Sprint name; empty when not specified.
    #[serde(default)]
    pub sprint: String,
}

/// Matches explicit Jira issue keys such as `PROJ-42` or `AB12-7`.
static ISSUE_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][A-Z0-9]+-\d+\b").unwrap());

/// Generic terms people use in speech/notes to refer to an issue by number
/// without its project prefix (e.g. "US 84", "story 84", "issue-84",
/// "user story 84"). Matched case-insensitively and combined with the
/// configured default project to build the real key (e.g. `CBMS-84`).
static IMPLICIT_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:user\s+story|us|story|issue|ticket|task|bug|epic)\s*#?\s*-?\s*(\d+)\b")
        .unwrap()
});

/// Bare prefixes (case-insensitive) that, even when they happen to match the
/// *explicit* key pattern (e.g. spoken/typed as `US-84`), mean "issue number
/// N in the default project" rather than a literal project code - "US" isn't
/// a real Jira project here, it's shorthand for "user story".
const GENERIC_PREFIXES: &[&str] = &["us", "story", "issue", "ticket", "task", "bug", "epic"];

/// Detect Jira issue keys mentioned in `transcript`, de-duplicated while
/// preserving first-seen order. Recognizes both explicit keys (`PROJ-42`) and
/// implicit references by number alone (`US 84`, `story 84`, `issue-84`, ...),
/// which are resolved against `default_project` (e.g. `US 84` + `CBMS` ->
/// `CBMS-84`). Implicit references are skipped when no default project is
/// configured, since there's nothing to resolve them against.
pub fn detect_issue_keys(transcript: &str, default_project: Option<&str>) -> Vec<String> {
    let mut seen = Vec::new();
    let mut push = |key: String| {
        if !seen.contains(&key) {
            seen.push(key);
        }
    };

    for m in ISSUE_KEY_RE.find_iter(transcript) {
        let raw = m.as_str();
        // A generic prefix (e.g. `US-84`) means "issue 84 in the default
        // project", not a literal project code - resolve it the same way as
        // the space-separated implicit form below.
        if let Some((prefix, number)) = raw.rsplit_once('-') {
            if GENERIC_PREFIXES.iter().any(|p| p.eq_ignore_ascii_case(prefix)) {
                if let Some(project) = default_project.filter(|p| !p.trim().is_empty()) {
                    push(format!("{}-{number}", project.trim()));
                    continue;
                }
            }
        }
        push(raw.to_string());
    }

    if let Some(project) = default_project.filter(|p| !p.trim().is_empty()) {
        for m in IMPLICIT_KEY_RE.captures_iter(transcript) {
            let number = &m[1];
            push(format!("{}-{number}", project.trim()));
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
- Output ONE proposal per distinct action item discussed. NEVER merge unrelated \
changes into a single proposal - e.g. an edit to an existing issue and the \
creation of a brand-new, different story are always two separate proposals, \
even if they were discussed back-to-back or in the same breath.\n\
- \"priority\", \"assignee\", and \"sprint\" are optional strings; \"labels\" is an \
optional array of strings. Leave them empty/omitted unless the meeting explicitly \
states them.\n\
- If nothing is actionable, output {{\"proposals\":[]}}.\n\n\
Example output:\n\
{{\"proposals\":[{{\"action\":\"create\",\"target_key\":\"\",\"issue_type\":\"Story\",\
\"summary\":\"Add password reset via email\",\"description\":\"Users request a reset \
link by email that expires after one hour.\",\"acceptance_criteria\":\"Link expires in 1h; \
one active link per user.\",\"rationale\":\"Team agreed users need self-service reset.\",\
\"priority\":\"High\",\"labels\":[\"auth\"],\"assignee\":\"\",\"sprint\":\"\"}}]}}\n\n\
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
        priority: get("priority"),
        labels: get_labels(value),
        assignee: get("assignee"),
        sprint: get("sprint"),
    })
}

/// Parse the `labels` field of a proposal, tolerating either a JSON array of
/// strings or a single comma-separated string (small models sometimes emit
/// either shape).
fn get_labels(value: &serde_json::Value) -> Vec<String> {
    match value.get("labels") {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(serde_json::Value::String(s)) => s
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => Vec::new(),
    }
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

/// Heuristic check for whether the raw LLM completion was cut off mid-JSON
/// (e.g. hit `max_new_tokens` before finishing the last proposal object).
///
/// Counts brace/bracket opens vs. closes across the whole string: a
/// well-formed completion always leaves every object/array it opened closed
/// (whether that's a full `{"proposals": [...]}` wrapper, or - when the prompt
/// was primed with the wrapper's opening - a sequence of self-contained
/// proposal objects). A surplus of opens means the last object was left
/// dangling, which is exactly what happens when generation is truncated
/// mid-proposal - and is otherwise silently dropped by [`parse_proposals`].
pub fn output_looks_truncated(raw: &str) -> bool {
    let mut brace = 0i32;
    let mut bracket = 0i32;
    for b in raw.bytes() {
        match b {
            b'{' => brace += 1,
            b'}' => brace -= 1,
            b'[' => bracket += 1,
            b']' => bracket -= 1,
            _ => {}
        }
    }
    brace > 0 || bracket > 0
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
        assert_eq!(detect_issue_keys(t, None), vec!["PROJ-42", "AB12-7"]);
    }

    #[test]
    fn detects_implicit_key_with_default_project() {
        let t = "Let's look at US 84 and add the acceptance criteria.";
        assert_eq!(detect_issue_keys(t, Some("CBMS")), vec!["CBMS-84"]);
    }

    #[test]
    fn detects_various_implicit_synonyms() {
        let t = "See story 12, issue 13, ticket-14, task 15, bug 16, epic 17, and user story 18.";
        assert_eq!(
            detect_issue_keys(t, Some("CBMS")),
            vec![
                "CBMS-12",
                "CBMS-13",
                "CBMS-14",
                "CBMS-15",
                "CBMS-16",
                "CBMS-17",
                "CBMS-18",
            ]
        );
    }

    #[test]
    fn implicit_key_ignored_without_default_project() {
        let t = "Let's look at US 84.";
        assert!(detect_issue_keys(t, None).is_empty());
    }

    #[test]
    fn generic_prefix_explicit_form_resolved_to_default_project() {
        // Spoken/typed as `US-84`: "US" here means "user story", not a literal
        // project code, so it should resolve against the default project.
        let t = "Update US-84 with the new points.";
        assert_eq!(detect_issue_keys(t, Some("CBMS")), vec!["CBMS-84"]);
    }

    #[test]
    fn real_project_key_not_confused_when_not_generic() {
        let t = "Update PROJ-42.";
        assert_eq!(detect_issue_keys(t, Some("CBMS")), vec!["PROJ-42"]);
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
            priority: String::new(),
            labels: Vec::new(),
            assignee: String::new(),
            sprint: String::new(),
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

    #[test]
    fn parses_new_fields_with_array_labels() {
        let raw = "{\"action\":\"create\",\"summary\":\"Add reset\",\"description\":\"via email\",\
\"priority\":\"High\",\"labels\":[\"auth\",\"backend\"],\"assignee\":\"alice\",\"sprint\":\"Sprint 4\"}";
        let props = parse_proposals(raw);
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].priority, "High");
        assert_eq!(props[0].labels, vec!["auth".to_string(), "backend".to_string()]);
        assert_eq!(props[0].assignee, "alice");
        assert_eq!(props[0].sprint, "Sprint 4");
    }

    #[test]
    fn parses_new_fields_with_comma_string_labels() {
        let raw = "{\"action\":\"create\",\"summary\":\"Add reset\",\"labels\":\"auth, backend ,\"}";
        let props = parse_proposals(raw);
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].labels, vec!["auth".to_string(), "backend".to_string()]);
    }

    #[test]
    fn output_truncated_flags_dangling_unclosed_object() {
        // Model was cut off mid-way through the second proposal's description.
        let raw = "{\"action\":\"create\",\"summary\":\"Add reset\"},\n\
{\"action\":\"create\",\"summary\":\"Second one\",\"description\":\"cut off here";
        assert!(output_looks_truncated(raw));
    }

    #[test]
    fn output_not_truncated_for_well_formed_wrapper() {
        let raw = "{\"proposals\":[{\"action\":\"create\",\"summary\":\"Add reset\"}]}";
        assert!(!output_looks_truncated(raw));
    }

    #[test]
    fn output_not_truncated_for_well_formed_primed_objects() {
        let raw = "{\"action\":\"create\",\"summary\":\"Add reset\"},\
{\"action\":\"update\",\"target_key\":\"PROJ-42\",\"summary\":\"Trim epic\"}";
        assert!(!output_looks_truncated(raw));
    }

    #[test]
    fn new_fields_default_to_empty_when_absent() {
        let raw = "{\"action\":\"create\",\"summary\":\"Add reset\"}";
        let props = parse_proposals(raw);
        assert_eq!(props.len(), 1);
        assert!(props[0].priority.is_empty());
        assert!(props[0].labels.is_empty());
        assert!(props[0].assignee.is_empty());
        assert!(props[0].sprint.is_empty());
    }
}
