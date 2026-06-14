//! Embedding backends behind a single trait.
//!
//! - [`bow::BowEmbedder`] — deterministic hashed bag-of-words. No deps, no
//!   network, no model. Always available; used for tests and offline fallback.
//! - `fast::FastEmbedder` — real bge-small / MiniLM via fastembed (ONNX). Behind
//!   the `fastembed` cargo feature.

pub mod bow;
#[cfg(feature = "fastembed")]
pub mod fast;

/// Whether a text is a search query or an indexed document. bge models are
/// asymmetric (query gets an instruction prefix); symmetric models ignore this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedKind {
    Query,
    Document,
}

pub trait Embedder {
    /// Stable id used as the index's `model` tag (changing it forces reindex).
    fn id(&self) -> String;
    fn embed(&self, texts: &[String], kind: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Pick a backend for `model`. With the `fastembed` feature and a recognized
/// model id, returns the real embedder; otherwise the offline bag-of-words one.
pub fn build(model: &str) -> anyhow::Result<Box<dyn Embedder>> {
    #[cfg(feature = "fastembed")]
    {
        if let Some(e) = fast::FastEmbedder::try_for(model)? {
            return Ok(Box::new(e));
        }
    }
    let _ = model;
    Ok(Box::new(bow::BowEmbedder::new()))
}
