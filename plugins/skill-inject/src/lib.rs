//! ski / skill-inject — local semantic auto-injection of agent skills.
//!
//! Milestone 1 surface: skill discovery, embedding index, and ranking
//! (`ski index` / `ski why`). The hook hot-path, model-load observation, and
//! `init` land in later milestones.

pub mod config;
pub mod embed;
pub mod index;
pub mod rank;
pub mod skill;
pub mod text;
