//! `ski` CLI. Milestones 1–2 implement `index`, `why`, and `hook`; `observe`
//! and `session-start` are stubbed until milestone 3.

use anyhow::Result;
use clap::{Parser, Subcommand};
use ski::config::Config;
use ski::embed::{self, EmbedKind};
use ski::hook::{self, Host};
use ski::index::{self, Index};
use ski::{paths, rank, skill};

#[derive(Parser)]
#[command(
    name = "ski",
    version,
    about = "skill-inject: local semantic skill auto-injection"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// (Re)build the persistent skill index.
    Index {
        /// Ignore the existing index and re-embed everything.
        #[arg(long)]
        rebuild: bool,
    },
    /// Rank skills against a prompt and print scores (tuning aid).
    Why {
        /// The prompt (all trailing words are joined).
        prompt: Vec<String>,
        /// How many ranked skills to show.
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// [stub, milestone 2] hook hot-path: decide + emit injection.
    Hook {
        #[arg(long)]
        host: String,
    },
    /// [stub, milestone 2] record skills the model loaded itself.
    Observe {
        #[arg(long)]
        host: String,
    },
    /// [stub, milestone 2] incremental reindex + re-arm session state.
    SessionStart {
        #[arg(long)]
        host: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::default();
    match cli.cmd {
        Cmd::Index { rebuild } => cmd_index(&cfg, rebuild),
        Cmd::Why { prompt, top } => cmd_why(&cfg, &prompt.join(" "), top),
        Cmd::Hook { host } => hook::run(host.parse::<Host>()?),
        Cmd::Observe { host } => stub("observe", &host),
        Cmd::SessionStart { host } => stub("session-start", &host),
    }
}

fn cmd_index(cfg: &Config, rebuild: bool) -> Result<()> {
    let index_path = paths::index_path();
    let skills = skill::discover(&cfg.roots)?;
    let embedder = embed::build(&cfg.model)?;
    let prev = if rebuild {
        None
    } else {
        Index::load(&index_path)?
    };
    let idx = index::build(&skills, embedder.as_ref(), prev.as_ref())?;
    idx.save(&index_path)?;
    println!(
        "indexed {} skills ({} dims) via '{}' -> {}",
        idx.skills.len(),
        idx.dim,
        idx.model,
        index_path.display()
    );
    Ok(())
}

fn cmd_why(cfg: &Config, prompt: &str, top: usize) -> Result<()> {
    let skills = skill::discover(&cfg.roots)?;
    if skills.is_empty() {
        println!("no skills found in roots: {:?}", cfg.roots);
        return Ok(());
    }
    let embedder = embed::build(&cfg.model)?;
    let idx = index::build(&skills, embedder.as_ref(), None)?;
    let query = embedder
        .embed(&[prompt.to_string()], EmbedKind::Query)?
        .remove(0);
    let hits = rank::rank_all(&query, prompt, &idx, cfg);

    println!(
        "embedder '{}'  threshold {:.2}  prompt: {prompt:?}",
        idx.model, cfg.min_similarity
    );
    for h in hits.iter().take(top) {
        let mark = if h.score >= cfg.min_similarity {
            "*"
        } else {
            " "
        };
        println!(
            "{mark} {:<26} score {:.3}  (cos {:.3} + kw {:.3})",
            h.name, h.score, h.cosine, h.keyword
        );
    }
    Ok(())
}

fn stub(name: &str, host: &str) -> Result<()> {
    eprintln!("ski {name}: not yet implemented (milestone 2); host={host}");
    Ok(())
}
