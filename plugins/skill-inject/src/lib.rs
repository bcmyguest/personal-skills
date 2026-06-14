//! ski / skill-inject — local semantic auto-injection of agent skills.
//!
//! Milestones 1–2 surface: skill discovery, embedding index, ranking
//! (`ski index` / `ski why`), and the hook hot-path with session dedup
//! (`ski hook`). Model-load observation and `init` land in later milestones.

pub mod config;
pub mod embed;
pub mod hook;
pub mod index;
pub mod inject;
pub mod paths;
pub mod rank;
pub mod session;
pub mod skill;
pub mod text;
