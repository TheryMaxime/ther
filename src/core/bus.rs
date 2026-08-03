//! The typed event bus that carries the "discussion" from the core services
//! (speech-to-text, the embedded LLM) outward to whatever UI a module ships.
//!
//! Every core service reports progress through a single cloneable
//! [`EventSender`] that emits [`CoreEvent`]s, replacing the previous set of
//! type-erased `Box<dyn Fn(String)>` callbacks. A module drains the matching
//! receiver and maps [`CoreEvent`]s onto its own UI state.

use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// A single update emitted by a core service.
///
/// The meaning is carried by the variant (not by *which* callback was invoked),
/// so the whole services ↔ UI ↔ LLM conversation flows over one typed channel.
#[derive(Clone, Debug)]
pub enum CoreEvent {
    /// Human-readable status/progress line.
    Status(String),
    /// The latest speech-to-text transcript snapshot.
    Transcript(String),
    /// The assistant's latest running-notes answer.
    Response(String),
}

/// Receiving half of the event bus, drained by a module's UI task.
pub type EventReceiver = UnboundedReceiver<CoreEvent>;

/// Cloneable, thread-safe sink handed to every core service. Sending never
/// blocks, so it is safe to call from audio/LLM worker threads.
#[derive(Clone)]
pub struct EventSender(UnboundedSender<CoreEvent>);

impl EventSender {
    /// Emit a raw [`CoreEvent`].
    pub fn send(&self, event: CoreEvent) {
        let _ = self.0.send(event);
    }

    /// Emit a [`CoreEvent::Status`].
    pub fn status(&self, text: impl Into<String>) {
        self.send(CoreEvent::Status(text.into()));
    }

    /// Emit a [`CoreEvent::Transcript`].
    pub fn transcript(&self, text: impl Into<String>) {
        self.send(CoreEvent::Transcript(text.into()));
    }

    /// Emit a [`CoreEvent::Response`].
    pub fn response(&self, text: impl Into<String>) {
        self.send(CoreEvent::Response(text.into()));
    }
}

/// Create a connected [`EventSender`] / [`EventReceiver`] pair.
pub fn channel() -> (EventSender, EventReceiver) {
    let (tx, rx) = unbounded_channel();
    (EventSender(tx), rx)
}
