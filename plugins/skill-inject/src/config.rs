//! Runtime configuration. Milestone 1 uses defaults only; a config-file loader
//! (`~/.config/ski/config.toml`) lands with the hook path in milestone 2.

use crate::embed::Embedder;
use std::path::PathBuf;

/// How a matched skill is delivered to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectMode {
    /// Tell the model a relevant skill exists and let it load the file (keeps
    /// model agency; the v1 default).
    Directive,
    /// Inject the `SKILL.md` body straight into context.
    Body,
}

/// Forcefulness of a `directive`-mode injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strength {
    /// Resolve from the host (Claude -> soft, opencode -> hard).
    Auto,
    /// A nudge — enough for a strong native chooser.
    Soft,
    /// An imperative — for weak local choosers.
    Hard,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Embedding model id. Recognized by the fastembed backend; otherwise the
    /// offline bag-of-words backend is used regardless of this value.
    pub model: String,
    /// Minimum hybrid score for a skill to be eligible for injection.
    pub min_similarity: f32,
    /// Max gap below the single best-scoring skill a skill may fall and still be
    /// injected. Suppresses the weak tail: when the top match is strong, only
    /// near-peers ride along; when only weak matches exist (or the leader was
    /// already injected this session), nothing clears the gate. Tuned alongside
    /// `min_similarity` per embedder.
    pub score_margin: f32,
    /// Max skills injected per prompt.
    pub max_skills: usize,
    /// Max total injected characters (budget; enforced in the hook path).
    pub char_budget: usize,
    /// Added to a skill's score per matching keyword.
    pub keyword_boost: f32,
    /// Filesystem roots scanned for `SKILL.md` files.
    pub roots: Vec<PathBuf>,
    /// How matched skills are injected.
    pub inject_mode: InjectMode,
    /// Forcefulness of directive-mode injections.
    pub directive_strength: Strength,
    /// Skill ids never auto-injected.
    pub deny: Vec<String>,
    /// Skill ids injected whenever a keyword hits, even below `min_similarity`.
    pub force: Vec<String>,

    // --- Stage-2 reranking (see `crate::rerank`). The thresholds below are on the
    // cross-encoder's logit scale, unrelated to the cosine thresholds above, and
    // are *not* touched by `calibrate_to`. ---
    /// Stage-1 score below which a prompt is treated as having no relevant skill,
    /// so the (costly) reranker is skipped entirely.
    pub recall_floor: f32,
    /// Stage-1 score above which the top match may be a confident lone winner.
    pub high_conf: f32,
    /// Minimum stage-1 gap from the top match to the runner-up for the top to
    /// count as a *lone* winner (and thus skip reranking).
    pub clear_gap: f32,
    /// How many stage-1 candidates are handed to the reranker.
    pub rerank_top_k: usize,
    /// Minimum reranker logit for a skill to be injected.
    pub rerank_min: f32,
    /// Max reranker-logit gap below the best reranked skill for a peer to ride along.
    pub rerank_margin: f32,
}

impl Config {
    /// Adopt the active embedder's score thresholds. Cosine distributions are a
    /// property of the embedding space, not user preference, so `min_similarity`
    /// and `score_margin` follow the embedder that actually ran (bge vs the
    /// offline bag-of-words fallback). Other fields are left untouched.
    pub fn calibrate_to(&mut self, embedder: &dyn Embedder) {
        self.min_similarity = embedder.min_similarity();
        self.score_margin = embedder.score_margin();
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "bge-small-en-v1.5".into(),
            min_similarity: 0.30,
            score_margin: 0.15,
            max_skills: 2,
            char_budget: 6000,
            keyword_boost: 0.15,
            roots: default_roots(),
            inject_mode: InjectMode::Directive,
            directive_strength: Strength::Auto,
            deny: Vec::new(),
            force: Vec::new(),
            // Reranker gate + thresholds, calibrated on the anthropic/skills
            // corpus against the JINA turbo reranker (see `examples/rerank_probe`).
            // Scoped top-1 accuracy: 76% stage-1 only -> 88% with reranking.
            //
            // `recall_floor` skips the reranker when nothing is plausibly relevant.
            // bge is anisotropic (unrelated prompts still cosine ~0.5), which
            // compresses the usable range: 0.50 skips clearly-irrelevant prompts
            // without dropping real-but-weak matches. `high_conf` is effectively
            // disabled (2.0): a confidence-based skip measurably *hurt* accuracy,
            // because the bi-encoder is confidently wrong on the confusable pairs
            // the reranker exists to fix. It is retained as a tunable, not removed.
            recall_floor: 0.50,
            high_conf: 2.0,
            clear_gap: 0.12,
            rerank_top_k: 12,
            rerank_min: -2.5,
            rerank_margin: 2.0,
        }
    }
}

fn default_roots() -> Vec<PathBuf> {
    // Opt-in override: colon-separated roots. Unset -> the defaults below.
    // Lets evals/tools scope discovery to one skill library without a config
    // file (e.g. `SKI_ROOTS=~/.claude/plugins/marketplaces/anthropic-agent-skills`).
    if let Some(raw) = std::env::var_os("SKI_ROOTS") {
        let roots: Vec<PathBuf> = std::env::split_paths(&raw)
            .filter(|p| !p.as_os_str().is_empty())
            .collect();
        if !roots.is_empty() {
            return roots;
        }
    }
    let mut v = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        v.push(home.join(".claude/skills"));
        v.push(home.join(".claude/plugins"));
        v.push(home.join(".config/opencode/skills"));
    }
    v.push(PathBuf::from(".claude/skills"));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{bow::BowEmbedder, EmbedKind, Embedder};

    /// Stands in for a dense embedder with its own (non-default) thresholds.
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn id(&self) -> String {
            "stub".into()
        }
        fn embed(&self, _: &[String], _: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }
        fn min_similarity(&self) -> f32 {
            0.64
        }
        fn score_margin(&self) -> f32 {
            0.12
        }
    }

    #[test]
    fn calibrate_adopts_embedder_thresholds() {
        let mut cfg = Config::default();
        cfg.calibrate_to(&StubEmbedder);
        assert_eq!(cfg.min_similarity, 0.64);
        assert_eq!(cfg.score_margin, 0.12);
    }

    #[test]
    fn calibrate_to_bow_uses_trait_defaults() {
        // The bag-of-words embedder doesn't override the trait defaults.
        let mut cfg = Config {
            min_similarity: 0.99,
            score_margin: 0.99,
            ..Default::default()
        };
        cfg.calibrate_to(&BowEmbedder::new());
        assert_eq!(cfg.min_similarity, 0.30);
        assert_eq!(cfg.score_margin, 0.15);
    }
}
