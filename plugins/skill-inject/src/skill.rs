//! Skill discovery and `SKILL.md` frontmatter parsing.

use crate::text::{fnv1a_64, tokenize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Skill {
    /// Unique id (the skill's declared `name`).
    pub id: String,
    pub name: String,
    pub description: String,
    /// Keywords for the hybrid keyword boost: explicit `keywords`/`aliases`
    /// frontmatter, plus tokens derived from the name.
    pub keywords: Vec<String>,
    pub path: PathBuf,
    /// Content hash for index cache invalidation.
    pub hash: String,
}

/// Walk `roots` and parse every `SKILL.md` found.
pub fn discover(roots: &[PathBuf]) -> anyhow::Result<Vec<Skill>> {
    let mut files = Vec::new();
    for r in roots {
        collect(r, &mut files);
    }
    files.sort();
    files.dedup();

    let mut out = Vec::new();
    for f in &files {
        if let Some(s) = parse_file(f)? {
            out.push(s);
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out.dedup_by(|a, b| a.id == b.id);
    Ok(out)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let skip = matches!(
                p.file_name().and_then(|s| s.to_str()),
                Some(".git" | "target" | "node_modules")
            );
            if !skip {
                collect(&p, out);
            }
        } else if p.file_name().and_then(|s| s.to_str()) == Some("SKILL.md") {
            out.push(p);
        }
    }
}

/// Parse a single `SKILL.md`. Returns `None` if it lacks a usable frontmatter.
pub fn parse_file(path: &Path) -> anyhow::Result<Option<Skill>> {
    let content = fs::read_to_string(path)?;
    let Some((name, description, mut keywords)) = parse_frontmatter(&content) else {
        return Ok(None);
    };
    if name.is_empty() || description.is_empty() {
        return Ok(None);
    }
    for tok in tokenize(&name) {
        if !keywords.contains(&tok) {
            keywords.push(tok);
        }
    }
    let hash = format!("{:016x}", fnv1a_64(content.as_bytes()));
    Ok(Some(Skill {
        id: name.clone(),
        name,
        description,
        keywords,
        path: path.to_path_buf(),
        hash,
    }))
}

/// Extract `name`, `description`, and `keywords`/`aliases` from a leading
/// `--- ... ---` YAML frontmatter block. Intentionally minimal: handles the
/// single-line `key: value` and inline-list shapes our skills use, not the full
/// YAML grammar (no block scalars / nested maps).
pub fn parse_frontmatter(content: &str) -> Option<(String, String, Vec<String>)> {
    let mut lines = content.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    let (mut name, mut description, mut keywords) = (String::new(), String::new(), Vec::new());
    for line in lines {
        let t = line.trim_end();
        if t.trim() == "---" {
            break;
        }
        if let Some(v) = t.strip_prefix("name:") {
            name = unquote(v.trim());
        } else if let Some(v) = t.strip_prefix("description:") {
            description = unquote(v.trim());
        } else if let Some(v) = t.strip_prefix("keywords:") {
            keywords = parse_list(v.trim());
        } else if let Some(v) = t.strip_prefix("aliases:") {
            keywords.extend(parse_list(v.trim()));
        }
    }
    Some((name, description, keywords))
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn parse_list(s: &str) -> Vec<String> {
    s.trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|x| unquote(x.trim()).to_ascii_lowercase())
        .filter(|x| !x.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_frontmatter() {
        let md = "---\nname: git-attribution\ndescription: Credit AI in commits.\n---\nbody\n";
        let (name, desc, _) = parse_frontmatter(md).unwrap();
        assert_eq!(name, "git-attribution");
        assert_eq!(desc, "Credit AI in commits.");
    }

    #[test]
    fn parses_quotes_and_keywords() {
        let md = "---\nname: \"x\"\ndescription: 'd'\nkeywords: [Foo, bar]\n---\n";
        let (name, desc, kw) = parse_frontmatter(md).unwrap();
        assert_eq!(name, "x");
        assert_eq!(desc, "d");
        assert_eq!(kw, ["foo", "bar"]);
    }

    #[test]
    fn rejects_without_frontmatter() {
        assert!(parse_frontmatter("no frontmatter here").is_none());
    }
}
