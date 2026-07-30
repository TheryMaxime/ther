//! Typed, async wrappers over the Atlassian Jira MCP tools.
//!
//! Wraps the low-level [`McpClient`] with Jira-specific calls, resolving and
//! caching the `cloudId` required by every Jira tool. All methods return plain
//! [`serde_json::Value`] payloads (already unwrapped from the MCP envelope) plus
//! a few convenience structs for the pieces the UI needs.

use anyhow::{anyhow, Context, Result};
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

    /// Create a new issue. Returns the raw create result (contains the new key).
    pub async fn create_issue(
        &self,
        project_key: &str,
        issue_type: &str,
        summary: &str,
        description: &str,
    ) -> Result<Value> {
        let cloud_id = self.cloud_id().await?;
        self.mcp
            .call_tool(
                "createJiraIssue",
                json!({
                    "cloudId": cloud_id,
                    "projectKey": project_key,
                    "issueTypeName": issue_type,
                    "summary": summary,
                    "description": description,
                }),
            )
            .await
            .with_context(|| format!("creating {issue_type} in {project_key}"))
    }

    /// Edit an existing issue's summary/description.
    pub async fn edit_issue(&self, key: &str, summary: &str, description: &str) -> Result<Value> {
        let cloud_id = self.cloud_id().await?;
        self.mcp
            .call_tool(
                "editJiraIssue",
                json!({
                    "cloudId": cloud_id,
                    "issueIdOrKey": key,
                    "fields": {
                        "summary": summary,
                        "description": description,
                    },
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

        // Create returns a new key.
        let created = client
            .create_issue("PROJ", "Story", "New story", "desc")
            .await
            .unwrap();
        assert_eq!(created_key(&created).as_deref(), Some("PROJ-1001"));

        // Edit + comment succeed.
        client.edit_issue("PROJ-57", "s", "d").await.unwrap();
        client.add_comment("PROJ-57", "note").await.unwrap();
    }
}
