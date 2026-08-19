//! Typed, async wrappers over the Atlassian Jira MCP tools.
//!
//! Wraps the low-level [`McpClient`] with Jira-specific calls, resolving and
//! caching the `cloudId` required by every Jira tool. All methods return plain
//! [`serde_json::Value`] payloads (already unwrapped from the MCP envelope) plus
//! a few convenience structs for the pieces the UI needs.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::core::config::JiraConfig;
use crate::core::mcp::McpClient;

/// A lightweight view of a Jira issue used for candidate confirmation.
#[derive(Clone, Debug)]
pub struct IssueSummary {
    pub key: String,
    pub summary: String,
    pub issue_type: String,
    pub status: String,
}

/// The extra (beyond summary/description) fields a create/update can carry.
/// All are optional; empty/blank values are omitted from the Jira request so
/// existing values aren't accidentally cleared.
#[derive(Clone, Debug, Default)]
pub struct IssueFields {
    pub priority: String,
    pub labels: Vec<String>,
    pub assignee: String,
    pub sprint: String,
}

/// A snapshot of an existing issue's current field values, used to render an
/// old→new diff for update proposals before they're applied.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IssueSnapshot {
    pub summary: String,
    pub description: String,
    pub priority: String,
    pub labels: Vec<String>,
    pub assignee: String,
    pub sprint: String,
}

/// Connected, Jira-aware MCP client with a cached cloudId.
pub struct JiraClient {
    mcp: McpClient,
    cloud_id: Mutex<Option<String>>,
    default_project: Option<String>,
}

impl JiraClient {
    /// Connect to the MCP server and prepare the Jira client.
    pub async fn connect(cfg: &JiraConfig) -> Result<Self> {
        let mcp = McpClient::connect(cfg)
            .await
            .context("connecting to Atlassian MCP server")?;
        Ok(Self {
            mcp,
            cloud_id: Mutex::new(cfg.cloud_id.clone()),
            default_project: cfg.default_project.clone(),
        })
    }

    pub fn default_project(&self) -> Option<&str> {
        self.default_project.as_deref()
    }

    /// Resolve (and cache) the Atlassian cloudId for the authenticated site.
    pub async fn cloud_id(&self) -> Result<String> {
        if let Some(id) = self.cloud_id.lock().await.clone() {
            return Ok(id);
        }

        let payload = self
            .mcp
            .call_tool("getAccessibleAtlassianResources", json!({}))
            .await
            .context("resolving accessible Atlassian resources")?;

        let id = payload
            .as_array()
            .and_then(|a| a.first())
            .and_then(|r| r.get("id"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("no accessible Atlassian cloud site found: {payload}"))?;

        *self.cloud_id.lock().await = Some(id.clone());
        Ok(id)
    }

    /// Search issues with JQL, returning lightweight summaries for confirmation.
    pub async fn search_jql(&self, jql: &str, max_results: u32) -> Result<Vec<IssueSummary>> {
        let cloud_id = self.cloud_id().await?;
        let payload = self
            .mcp
            .call_tool(
                "searchJiraIssuesUsingJql",
                json!({
                    "cloudId": cloud_id,
                    "jql": jql,
                    "maxResults": max_results,
                    "fields": ["summary", "issuetype", "status"],
                }),
            )
            .await
            .with_context(|| format!("JQL search failed: {jql}"))?;

        Ok(parse_issue_list(&payload))
    }

    /// Fetch a single issue's full detail by key (e.g. `PROJ-42`).
    pub async fn get_issue(&self, key: &str) -> Result<Value> {
        let cloud_id = self.cloud_id().await?;
        self.mcp
            .call_tool(
                "getJiraIssue",
                json!({ "cloudId": cloud_id, "issueIdOrKey": key }),
            )
            .await
            .with_context(|| format!("fetching issue {key}"))
    }

    /// Fetch and parse an issue's current field values, for diffing against a
    /// proposed update before it's applied.
    pub async fn get_issue_fields_snapshot(&self, key: &str) -> Result<IssueSnapshot> {
        let payload = self.get_issue(key).await?;
        Ok(parse_issue_snapshot(&payload))
    }

    /// Create a new issue. Returns the raw create result (contains the new key).
    pub async fn create_issue(
        &self,
        project_key: &str,
        issue_type: &str,
        summary: &str,
        description: &str,
        fields: &IssueFields,
    ) -> Result<Value> {
        let cloud_id = self.cloud_id().await?;
        let mut args = json!({
            "cloudId": cloud_id,
            "projectKey": project_key,
            "issueTypeName": issue_type,
            "summary": summary,
            "description": description,
        });
        merge_extra_fields(&mut args, fields);
        self.mcp
            .call_tool("createJiraIssue", args)
            .await
            .with_context(|| format!("creating {issue_type} in {project_key}"))
    }

    /// Edit an existing issue's summary/description plus any optional extra
    /// fields (priority/labels/assignee/sprint) that were provided.
    pub async fn edit_issue(
        &self,
        key: &str,
        summary: &str,
        description: &str,
        fields: &IssueFields,
    ) -> Result<Value> {
        let cloud_id = self.cloud_id().await?;
        let mut field_obj = json!({
            "summary": summary,
            "description": description,
        });
        merge_extra_fields(&mut field_obj, fields);
        self.mcp
            .call_tool(
                "editJiraIssue",
                json!({
                    "cloudId": cloud_id,
                    "issueIdOrKey": key,
                    "fields": field_obj,
                }),
            )
            .await
            .with_context(|| format!("editing issue {key}"))
    }

    /// Add a comment to an existing issue.
    pub async fn add_comment(&self, key: &str, body: &str) -> Result<Value> {
        let cloud_id = self.cloud_id().await?;
        self.mcp
            .call_tool(
                "addCommentToJiraIssue",
                json!({
                    "cloudId": cloud_id,
                    "issueIdOrKey": key,
                    "commentBody": body,
                }),
            )
            .await
            .with_context(|| format!("commenting on issue {key}"))
    }
}

/// Extract the `key` of a newly created issue from a create result.
pub fn created_key(value: &Value) -> Option<String> {
    value
        .get("key")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Merge non-empty [`IssueFields`] into a Jira `fields`-shaped JSON object,
/// omitting blank values so existing Jira data isn't accidentally cleared.
fn merge_extra_fields(target: &mut Value, fields: &IssueFields) {
    let Some(obj) = target.as_object_mut() else { return };
    if !fields.priority.trim().is_empty() {
        obj.insert("priority".to_string(), json!({ "name": fields.priority }));
    }
    if !fields.labels.is_empty() {
        obj.insert("labels".to_string(), json!(fields.labels));
    }
    if !fields.assignee.trim().is_empty() {
        obj.insert("assignee".to_string(), json!({ "name": fields.assignee }));
    }
    if !fields.sprint.trim().is_empty() {
        // Sprint is a per-instance custom field in real Jira; the MCP server
        // is expected to resolve it by name via this generic key.
        obj.insert("sprint".to_string(), json!(fields.sprint));
    }
}

/// Parse a `getJiraIssue` payload into an [`IssueSnapshot`] of current values.
fn parse_issue_snapshot(payload: &Value) -> IssueSnapshot {
    let fields = payload.get("fields");
    let str_field = |path: &[&str]| -> String {
        let mut node = fields;
        for p in path {
            node = node.and_then(|n| n.get(*p));
        }
        node.and_then(Value::as_str).unwrap_or("").to_string()
    };
    let labels = fields
        .and_then(|f| f.get("labels"))
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default();
    IssueSnapshot {
        summary: str_field(&["summary"]),
        description: str_field(&["description"]),
        priority: str_field(&["priority", "name"]),
        labels,
        assignee: str_field(&["assignee", "name"]),
        sprint: str_field(&["sprint"]),
    }
}

/// Parse a `searchJiraIssuesUsingJql` payload into [`IssueSummary`] rows.
fn parse_issue_list(payload: &Value) -> Vec<IssueSummary> {
    let Some(issues) = payload.get("issues").and_then(Value::as_array) else {
        return Vec::new();
    };
    issues
        .iter()
        .filter_map(|issue| {
            let key = issue.get("key").and_then(Value::as_str)?.to_string();
            let fields = issue.get("fields");
            let field = |path: &[&str]| -> String {
                let mut node = fields;
                for p in path {
                    node = node.and_then(|n| n.get(*p));
                }
                node.and_then(Value::as_str).unwrap_or("").to_string()
            };
            Some(IssueSummary {
                key,
                summary: field(&["summary"]),
                issue_type: field(&["issuetype", "name"]),
                status: field(&["status", "name"]),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_cfg() -> JiraConfig {
        JiraConfig {
            base_url: "http://mock".to_string(),
            email: String::new(),
            api_token: String::new(),
            default_project: Some("PROJ".to_string()),
            cloud_id: None,
            mock: true,
        }
    }

    #[tokio::test]
    async fn mock_client_full_flow() {
        let client = JiraClient::connect(&mock_cfg()).await.unwrap();

        // cloudId is resolved and cached.
        assert_eq!(client.cloud_id().await.unwrap(), "mock-cloud-id");

        // JQL search returns typed summaries.
        let hits = client.search_jql("text ~ \"onboarding\"", 5).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].key, "PROJ-42");
        assert_eq!(hits[0].issue_type, "Epic");

        // Single issue fetch echoes the requested key.
        let issue = client.get_issue("PROJ-57").await.unwrap();
        assert_eq!(issue.get("key").and_then(|v| v.as_str()), Some("PROJ-57"));

        // Snapshot parses current field values for diffing.
        let snapshot = client.get_issue_fields_snapshot("PROJ-57").await.unwrap();
        assert_eq!(snapshot.priority, "Medium");
        assert_eq!(snapshot.labels, vec!["onboarding".to_string()]);
        assert_eq!(snapshot.assignee, "Bob");

        // Create returns a new key.
        let created = client
            .create_issue("PROJ", "Story", "New story", "desc", &IssueFields::default())
            .await
            .unwrap();
        assert_eq!(created_key(&created).as_deref(), Some("PROJ-1001"));

        // Create with extra fields still succeeds (mock ignores request shape).
        let fields = IssueFields {
            priority: "High".to_string(),
            labels: vec!["auth".to_string()],
            assignee: "alice".to_string(),
            sprint: "Sprint 4".to_string(),
        };
        client
            .create_issue("PROJ", "Story", "New story", "desc", &fields)
            .await
            .unwrap();

        // Edit + comment succeed, with and without extra fields.
        client.edit_issue("PROJ-57", "s", "d", &IssueFields::default()).await.unwrap();
        client.edit_issue("PROJ-57", "s", "d", &fields).await.unwrap();
        client.add_comment("PROJ-57", "note").await.unwrap();
    }
}
