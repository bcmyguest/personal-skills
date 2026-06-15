//! Throwaway probe: does a cross-encoder reranker fix the bi-encoder's misses?
//!
//! Stage 1 (bge-small) retrieves top-K candidates; stage 2 (a fastembed
//! reranker) re-scores (prompt, description+body_head) pairs. For each hard
//! prompt we print the bi-encoder's top-1 and the reranker's top-1 so we can see
//! whether reranking promotes the gold skill and separates negatives.
//!
//!   SKI_ROOTS=~/.claude/plugins/marketplaces/anthropic-agent-skills \
//!     cargo run --example rerank_probe --features fastembed
//!
//! Not wired into the product; delete once the decision is made.

#[cfg(not(feature = "fastembed"))]
fn main() {
    eprintln!("run with --features fastembed");
}

#[cfg(feature = "fastembed")]
fn main() -> anyhow::Result<()> {
    use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
    use ski::config::Config;
    use ski::embed::{self, EmbedKind};
    use ski::index;
    use ski::rank;
    use ski::skill;

    // (prompt, expected gold skill) — the scoped/global misses, plus negatives.
    let cases: &[(&str, &str)] = &[
        ("create an original piece of visual art and save it as a PDF", "canvas-design"),
        ("I need a printable art piece to hang on the wall", "canvas-design"),
        ("how do I stream tool use with the Anthropic SDK and count tokens", "claude-api"),
        ("draft an incident report for last night's outage", "internal-comms"),
        ("I want to expose this external service to an LLM through tools", "mcp-builder"),
        ("turn the data in this spreadsheet into a slide deck", "xlsx"),
        ("export this Word report to a PDF", "docx"),
        ("verify the frontend of the app I just started works", "webapp-testing"),
        ("make this slide match our company's visual identity", "brand-guidelines"),
        // negatives: gold is (none) — reranker should score the top hit LOW.
        ("interactively rebase to squash the last three git commits", "(none)"),
        ("explain how the TCP three-way handshake works", "(none)"),
        ("rename the variable foo to bar in this rust function", "(none)"),
    ];

    const TOP_K: usize = 12; // raise from 8: claude-api fell out of top-8 recall.

    let cfg = Config::default();
    let skills = skill::discover(&cfg.roots)?;
    eprintln!("discovered {} skills in {:?}", skills.len(), cfg.roots);
    let embedder = embed::build(&cfg.model)?;
    let idx = index::build(&skills, embedder.as_ref(), None)?;

    let body_of = |id: &str| -> String {
        skills
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.doc_text())
            .unwrap_or_default()
    };

    // Precompute stage-1 candidates once; both rerankers see the same input.
    struct Case {
        prompt: String,
        gold: String,
        cand_ids: Vec<String>,
        cand_names: Vec<String>,
        docs: Vec<String>,
    }
    let mut prepared = Vec::new();
    for (prompt, gold) in cases {
        let q = embedder.embed(&[prompt.to_string()], EmbedKind::Query)?.remove(0);
        let ranked = rank::rank_all(&q, prompt, &idx, &cfg);
        let cands: Vec<&rank::Hit> = ranked.iter().take(TOP_K).collect();
        prepared.push(Case {
            prompt: prompt.to_string(),
            gold: gold.to_string(),
            cand_ids: cands.iter().map(|h| h.id.clone()).collect(),
            cand_names: cands.iter().map(|h| h.name.clone()).collect(),
            docs: cands.iter().map(|h| body_of(&h.id)).collect(),
        });
    }

    let models = [
        ("turbo", RerankerModel::JINARerankerV1TurboEn),
        ("bge-base", RerankerModel::BGERerankerBase),
    ];
    for (label, model) in models {
        eprintln!("\n=== reranker: {label} ===");
        let t_load = std::time::Instant::now();
        let reranker = TextRerank::try_new(
            RerankInitOptions::new(model).with_show_download_progress(false),
        )?;
        let load_ms = t_load.elapsed().as_millis();
        // Single-prompt rerank latency (the real hot-path cost per UserPromptSubmit).
        let warm = &prepared[0];
        let t1 = std::time::Instant::now();
        let _ = reranker.rerank(warm.prompt.clone(), warm.docs.clone(), false, None)?;
        let infer_ms = t1.elapsed().as_millis();
        eprintln!("  {label}: model load {load_ms} ms | one rerank ({} docs) {infer_ms} ms", warm.docs.len());
        let mut hits = 0usize;
        let mut min_pos = f32::INFINITY; // lowest score among correct-top1 positives
        let mut max_neg = f32::NEG_INFINITY; // highest score among negatives
        for c in &prepared {
            let rr = reranker.rerank(c.prompt.clone(), c.docs.clone(), false, None)?;
            let best = &rr[0];
            let top = &c.cand_names[best.index];
            let is_neg = c.gold == "(none)";
            let want = if is_neg { "" } else { c.gold.as_str() };
            let ok = top == want;
            let recall = c.cand_ids.iter().any(|id| id == &c.gold);
            if ok {
                hits += 1;
            }
            if is_neg {
                max_neg = max_neg.max(best.score);
            } else if ok {
                min_pos = min_pos.min(best.score);
            }
            println!(
                "  {:<55} -> {:<20} {:+.2}{}{}",
                &c.prompt[..c.prompt.len().min(54)],
                top,
                best.score,
                if ok { " OK" } else { "" },
                if !recall && !is_neg { " [recall-miss]" } else { "" },
            );
        }
        println!(
            "  {label}: top1 {hits}/{}  | separation: min_pos {:+.2} vs max_neg {:+.2}  (gap {:+.2})",
            prepared.len(),
            min_pos,
            max_neg,
            min_pos - max_neg,
        );
    }
    Ok(())
}
