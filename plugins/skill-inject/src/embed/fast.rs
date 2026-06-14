//! Real embeddings via fastembed (ONNX). Compiled only with the `fastembed`
//! feature. Default: bge-small-en-v1.5; lite alt: all-MiniLM-L6-v2 (quantized).
//!
//! NOTE: the fastembed 4.x API surface (InitOptions / embed signatures) may
//! need a minor tweak when the feature is first built against the pinned
//! version; this is wired but not exercised in the offline default build.

use crate::embed::{EmbedKind, Embedder};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

pub struct FastEmbedder {
    model: TextEmbedding,
    tag: String,
    bge: bool,
}

impl FastEmbedder {
    /// `Some` if `model` is a recognized fastembed model id, else `None` so the
    /// caller can fall back to the bag-of-words embedder.
    pub fn try_for(model: &str) -> anyhow::Result<Option<Self>> {
        let (em, bge) = match model {
            "bge-small-en-v1.5" => (EmbeddingModel::BGESmallENV15, true),
            "all-MiniLM-L6-v2-q" => (EmbeddingModel::AllMiniLML6V2Q, false),
            "all-MiniLM-L6-v2" => (EmbeddingModel::AllMiniLML6V2, false),
            _ => return Ok(None),
        };
        let te = TextEmbedding::try_new(InitOptions::new(em))?;
        Ok(Some(Self {
            model: te,
            tag: model.to_string(),
            bge,
        }))
    }
}

impl Embedder for FastEmbedder {
    fn id(&self) -> String {
        self.tag.clone()
    }

    fn embed(&self, texts: &[String], kind: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>> {
        let prepped: Vec<String> = if self.bge && kind == EmbedKind::Query {
            texts
                .iter()
                .map(|t| format!("{BGE_QUERY_PREFIX}{t}"))
                .collect()
        } else {
            texts.to_vec()
        };
        Ok(self.model.embed(prepped, None)?)
    }
}
