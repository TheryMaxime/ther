//! Minimal Model Context Protocol (MCP) client over **streamable HTTP**, used to
//! talk to the Atlassian Rovo MCP server with API-token (Basic) authentication.
//!
//! Only the subset we need is implemented: the `initialize` handshake and
//! `tools/call`. Responses are accepted as either `application/json` or
//! `text/event-stream` (SSE). Set `JIRA_MCP_MOCK=1` to return canned data without
//! any network access (used for offline development and verification).

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde_json::{json, Value};

use crate::core::config::JiraConfig;

const PROTOCOL_VERSION: &str = "2025-06-18";

/// A connected MCP client.
pub struct McpClient {
    http: reqwest::Client,
    url: String,
    auth_header: String,
    session_id: Mutex<Option<String>>,
    next_id: AtomicI64,
    mock: bool,
}

impl McpClient {
    /// Connect and perform the MCP `initialize` handshake (skipped in mock mode).
    pub async fn connect(cfg: &JiraConfig) -> Result<Self> {
        let creds = format!("{}:{}", cfg.email, cfg.api_token);
        let auth_header = format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode(creds)
        );

        let client = Self {
            http: reqwest::Client::new(),
            url: cfg.base_url.clone(),
            auth_header,
            session_id: Mutex::new(None),
            next_id: AtomicI64::new(1),
            mock: cfg.mock,
        };

        if !client.mock {
            client.initialize().await?;
        }
        Ok(client)
    }

    fn next_id(&self) -> i64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    async fn initialize(&self) -> Result<()> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "ai4jira", "version": env!("CARGO_PKG_VERSION") }
        });
        self.send_request("initialize", params)
            .await
            .context("MCP initialize failed")?;
        // Tell the server we are ready. Notifications carry no id and no response.
        self.send_notification("notifications/initialized", json!({}))
            .await?;
        Ok(())
    }

    /// Call an MCP tool and return the tool result payload.
    ///
    /// Atlassian tools return their data as a JSON string inside the first text
    /// content block; we parse that back into a [`Value`] when possible.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        if self.mock {
            return Ok(mock_tool(name, &arguments));
        }

        let params = json!({ "name": name, "arguments": arguments });
        let result = self
            .send_request("tools/call", params)
            .await
            .with_context(|| format!("MCP tools/call '{name}' failed"))?;

        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            bail!("MCP tool '{name}' returned an error: {result}");
        }

        Ok(extract_tool_payload(&result))
    }

    async fn send_notification(&self, method: &str, params: Value) -> Result<()> {
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        self.post(&body).await?;
        Ok(())
    }

    /// Send a JSON-RPC request and return its `result` value.
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id();
        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let text = self.post(&body).await?;
        let message = parse_response(&text, id)?;

        if let Some(err) = message.get("error") {
            bail!("MCP error: {err}");
        }
        message
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow!("MCP response missing 'result': {message}"))
    }

    /// POST a JSON-RPC body and return the raw response text (JSON or SSE).
    async fn post(&self, body: &Value) -> Result<String> {
        let mut req = self
            .http
            .post(&self.url)
            .header("Authorization", &self.auth_header)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", PROTOCOL_VERSION);

        if let Some(sid) = self.session_id.lock().unwrap().clone() {
            req = req.header("Mcp-Session-Id", sid);
        }

        let resp = req
            .json(body)
            .send()
            .await
            .context("MCP HTTP request failed")?;

        // Capture a session id handed back by the server (usually on initialize).
        if let Some(sid) = resp.headers().get("Mcp-Session-Id") {
            if let Ok(sid) = sid.to_str() {
                *self.session_id.lock().unwrap() = Some(sid.to_string());
            }
        }

        let status = resp.status();
        let text = resp.text().await.context("reading MCP response body")?;
        if !status.is_success() {
            bail!("MCP HTTP {status}: {text}");
        }
        Ok(text)
    }
}

/// Parse a streamable-HTTP response body (plain JSON or SSE) into the JSON-RPC
/// message whose `id` matches `want_id`.
fn parse_response(text: &str, want_id: i64) -> Result<Value> {
    let trimmed = text.trim_start();

    // Plain JSON response.
    if trimmed.starts_with('{') {
        return serde_json::from_str(trimmed).context("invalid JSON-RPC response");
    }

    // SSE: gather `data:` payloads and pick the matching JSON-RPC response.
    let mut fallback: Option<Value> = None;
    for line in text.lines() {
        let line = line.trim_start();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(data) {
            if value.get("id").and_then(Value::as_i64) == Some(want_id) {
                return Ok(value);
            }
            fallback.get_or_insert(value);
        }
    }

    fallback.ok_or_else(|| anyhow!("no JSON-RPC message found in MCP response: {text}"))
}

/// Pull the useful payload out of an MCP tool result. Atlassian tools embed a
/// JSON string in `content[0].text`; parse it, else fall back to raw text/result.
fn extract_tool_payload(result: &Value) -> Value {
    if let Some(content) = result.get("content").and_then(Value::as_array) {
        for block in content {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    return serde_json::from_str::<Value>(text)
                        .unwrap_or_else(|_| Value::String(text.to_string()));
                }
            }
        }
    }
    // Some servers place structured output here.
    if let Some(structured) = result.get("structuredContent") {
        return structured.clone();
    }
    result.clone()
}

/// Canned responses for `JIRA_MCP_MOCK=1` so the full flow works offline.
fn mock_tool(name: &str, arguments: &Value) -> Value {
    match name {
        "getAccessibleAtlassianResources" => json!([
            { "id": "mock-cloud-id", "name": "Mock Site", "url": "https://mock.atlassian.net" }
        ]),
        "searchJiraIssuesUsingJql" => json!({
            "issues": [
                { "key": "PROJ-42", "fields": {
                    "summary": "Epic: Onboarding revamp",
                    "issuetype": { "name": "Epic" },
                    "status": { "name": "In Progress" } } },
                { "key": "PROJ-57", "fields": {
                    "summary": "As a user I can reset my password",
                    "issuetype": { "name": "Story" },
                    "status": { "name": "To Do" } } }
            ]
        }),
        "getJiraIssue" => {
            let key = arguments
                .get("issueIdOrKey")
                .and_then(Value::as_str)
                .unwrap_or("PROJ-57");
            json!({
                "key": key,
                "fields": {
                    "summary": "As a user I can reset my password",
                    "description": "Existing story under the onboarding epic.",
                    "issuetype": { "name": "Story" },
                    "status": { "name": "To Do" }
                }
            })
        }
        "createJiraIssue" => json!({ "key": "PROJ-1001", "id": "1001",
            "self": "https://mock.atlassian.net/rest/api/3/issue/1001" }),
        "editJiraIssue" => json!({ "success": true }),
        "addCommentToJiraIssue" => json!({ "id": "20001", "success": true }),
        "getJiraProjectIssueTypesMetadata" => json!({
            "issueTypes": [ { "name": "Epic" }, { "name": "Story" }, { "name": "Task" }, { "name": "Bug" } ]
        }),
        other => json!({ "mock": true, "tool": other }),
    }
}
