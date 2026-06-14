//! Runtime configuration. Milestone 1 uses defaults only; a config-file loader
//! (`~/.config/ski/config.toml`) lands with the hook path in milestone 2.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    /// Embedding model id. Recognized by the fastembed backend; otherwise the
    /// offline bag-of-words backend is used regardless of this value.
    pub model: String,
    /// Minimum hybrid score for a skill to be eligible for injection.
    pub min_similarity: f32,
    /// Max skills injected per prompt.
    pub max_skills: usize,
    /// Max total injected characters (budget; enforced in the hook path).
    pub char_budget: usize,
    /// Added to a skill's score per matching keyword.
    pub keyword_boost: f32,
    /// Filesystem roots scanned for `SKILL.md` files.
    pub roots: Vec<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "bge-small-en-v1.5".into(),
            min_similarity: 0.30,
            max_skills: 2,
            char_budget: 6000,
            keyword_boost: 0.15,
            roots: default_roots(),
        }
    }
}

fn default_roots() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        v.push(home.join(".claude/skills"));
        v.push(home.join(".claude/plugins"));
        v.push(home.join(".config/opencode/skills"));
    }
    v.push(PathBuf::from(".claude/skills"));
    v
}
