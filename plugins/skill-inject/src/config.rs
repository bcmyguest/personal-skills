//! Runtime configuration. Milestone 1 uses defaults only; a config-file loader
//! (`~/.config/ski/config.toml`) lands with the hook path in milestone 2.

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
