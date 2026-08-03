//! Generic, in-memory context-saving cache shared by all modules.
//!
//! Modules often need to remember structured data across actions within a
//! single run (e.g. the Jira module's reviewable proposals) so later work can
//! *fetch* what was previously produced and decide whether to update it in
//! place or create something new, instead of always starting from scratch.
//!
//! [`ContextStore`] is a small, cheap-to-clone handle (same pattern as
//! [`crate::core::EventSender`]) around a `HashMap` keyed by
//! `(module, collection, id)`, storing arbitrary JSON-serializable values.
//! It intentionally holds everything in memory only: entries live for the
//! process lifetime and are lost on restart. Nothing here precludes adding a
//! disk-backed implementation later; callers only depend on this API.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

/// Composite key a context entry is stored under: `(module, collection, id)`.
type Key = (String, String, String);

/// Cheap-clone, thread-safe in-memory cache for arbitrary module context.
///
/// Some methods here (`get`, `get_raw`, `remove`, `update`) aren't exercised
/// by the Jira module yet (it only needs `put`/`list`), but are part of the
/// generic API any future module can rely on — kept `#[allow(dead_code)]`
/// rather than trimmed, since this is intentionally reusable infrastructure.
#[derive(Clone, Default)]
pub struct ContextStore {
    inner: Arc<Mutex<HashMap<Key, Value>>>,
}

#[allow(dead_code)]
impl ContextStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store (insert or overwrite) a value under `(module, collection, id)`.
    pub fn put<T: Serialize>(
        &self,
        module: &str,
        collection: &str,
        id: impl Into<String>,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        let json = serde_json::to_value(value)?;
        self.put_raw(module, collection, id, json);
        Ok(())
    }

    /// Store a pre-built [`Value`] directly, skipping serialization.
    pub fn put_raw(&self, module: &str, collection: &str, id: impl Into<String>, value: Value) {
        let key = (module.to_string(), collection.to_string(), id.into());
        self.inner.lock().unwrap().insert(key, value);
    }

    /// Fetch and deserialize a value, if present and it deserializes cleanly.
    pub fn get<T: DeserializeOwned>(&self, module: &str, collection: &str, id: &str) -> Option<T> {
        self.get_raw(module, collection, id)
            .and_then(|v| serde_json::from_value(v).ok())
    }

    /// Fetch the raw [`Value`] for a key, without deserializing.
    pub fn get_raw(&self, module: &str, collection: &str, id: &str) -> Option<Value> {
        let key = (module.to_string(), collection.to_string(), id.to_string());
        self.inner.lock().unwrap().get(&key).cloned()
    }

    /// List every `(id, value)` pair in a `(module, collection)`, silently
    /// skipping entries that fail to deserialize into `T`.
    pub fn list<T: DeserializeOwned>(&self, module: &str, collection: &str) -> Vec<(String, T)> {
        let guard = self.inner.lock().unwrap();
        guard
            .iter()
            .filter(|((m, c, _), _)| m == module && c == collection)
            .filter_map(|((_, _, id), v)| {
                serde_json::from_value::<T>(v.clone())
                    .ok()
                    .map(|value| (id.clone(), value))
            })
            .collect()
    }

    /// Remove and return a stored raw value, if present.
    pub fn remove(&self, module: &str, collection: &str, id: &str) -> Option<Value> {
        let key = (module.to_string(), collection.to_string(), id.to_string());
        self.inner.lock().unwrap().remove(&key)
    }

    /// Fetch-or-default, mutate in place, and store back — all under a single
    /// lock acquisition (via `get`/`put`, which each take the lock briefly).
    pub fn update<T, F>(
        &self,
        module: &str,
        collection: &str,
        id: impl Into<String>,
        default: impl FnOnce() -> T,
        f: F,
    ) where
        T: Serialize + DeserializeOwned,
        F: FnOnce(&mut T),
    {
        let id = id.into();
        let mut value = self.get::<T>(module, collection, &id).unwrap_or_else(default);
        f(&mut value);
        let _ = self.put(module, collection, id, &value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct Widget {
        name: String,
        count: u32,
    }

    #[test]
    fn put_get_roundtrip() {
        let store = ContextStore::new();
        let w = Widget { name: "gizmo".into(), count: 3 };
        store.put("jira", "widgets", "1", &w).unwrap();
        let got: Option<Widget> = store.get("jira", "widgets", "1");
        assert_eq!(got, Some(w));
    }

    #[test]
    fn get_missing_is_none() {
        let store = ContextStore::new();
        let got: Option<Widget> = store.get("jira", "widgets", "missing");
        assert_eq!(got, None);
    }

    #[test]
    fn list_scoped_to_module_and_collection() {
        let store = ContextStore::new();
        store
            .put("jira", "widgets", "1", &Widget { name: "a".into(), count: 1 })
            .unwrap();
        store
            .put("jira", "widgets", "2", &Widget { name: "b".into(), count: 2 })
            .unwrap();
        // Different collection / module: must not leak into the list below.
        store
            .put("jira", "other", "1", &Widget { name: "c".into(), count: 9 })
            .unwrap();
        store
            .put("other-module", "widgets", "1", &Widget { name: "d".into(), count: 9 })
            .unwrap();

        let mut items = store.list::<Widget>("jira", "widgets");
        items.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0, "1");
        assert_eq!(items[0].1.name, "a");
        assert_eq!(items[1].0, "2");
        assert_eq!(items[1].1.name, "b");
    }

    #[test]
    fn remove_deletes_entry() {
        let store = ContextStore::new();
        store
            .put("jira", "widgets", "1", &Widget { name: "a".into(), count: 1 })
            .unwrap();
        let removed = store.remove("jira", "widgets", "1");
        assert!(removed.is_some());
        assert_eq!(store.get::<Widget>("jira", "widgets", "1"), None);
        assert_eq!(store.remove("jira", "widgets", "1"), None);
    }

    #[test]
    fn update_with_default_and_mutation() {
        let store = ContextStore::new();
        store.update(
            "jira",
            "widgets",
            "1",
            || Widget { name: "fresh".into(), count: 0 },
            |w| w.count += 1,
        );
        let got: Widget = store.get("jira", "widgets", "1").unwrap();
        assert_eq!(got, Widget { name: "fresh".into(), count: 1 });

        // Second call should fetch the existing value (not the default) and
        // mutate on top of it.
        store.update(
            "jira",
            "widgets",
            "1",
            || Widget { name: "should-not-be-used".into(), count: 100 },
            |w| w.count += 1,
        );
        let got: Widget = store.get("jira", "widgets", "1").unwrap();
        assert_eq!(got, Widget { name: "fresh".into(), count: 2 });
    }
}
