#!/usr/bin/env python3
"""Render skill chains from chains/skills.dot and the skill sources.

Writes CHAINS.md, the README "Skill chains" section, and chains/teams/*.dot.
--check fails on stale output or an inconsistent registry; --wteam also runs
`wteam graph --team` on every DOT file when wteam is installed.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "chains" / "skills.dot"
TEAMS_DIR = ROOT / "chains" / "teams"
SKILLS_DIR = ROOT / "skills"
CHAINS_MD = ROOT / "CHAINS.md"
README = ROOT / "README.md"
README_START = "<!-- chains:start -->"
README_END = "<!-- chains:end -->"

NODE_TYPES = ("skill", "external", "action")
RELS = ("prerequisite", "transition", "suggested", "alternative")
ORIGINS = ("observed", "inferred")
SKILL_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NEXT_ITEM = re.compile(r"^- `([a-z0-9-]+)` \(([a-z]+)\)")
TEAM_KEY = "team:"
ARROWS = {
    "prerequisite": "==>",
    "transition": "-->",
    "suggested": "-.->",
    "alternative": "--o",
}


class RegistryError(Exception):
    pass


# --- DOT subset (the one wteam parses) -------------------------------------------


def tokenize(src):
    toks, i, line = [], 0, 1
    while i < len(src):
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif src.startswith("//", i) or c == "#":
            i = src.find("\n", i)
            i = len(src) if i < 0 else i
        elif src.startswith("/*", i):
            end = src.find("*/", i + 2)
            if end < 0:
                raise RegistryError(f"line {line}: unterminated /* comment")
            line += src.count("\n", i, end)
            i = end + 2
        elif c == '"':
            j, buf = i + 1, []
            while j < len(src) and src[j] != '"':
                if src[j] == "\\" and j + 1 < len(src) and src[j + 1] == '"':
                    j += 1
                buf.append(src[j])
                j += 1
            if j >= len(src):
                raise RegistryError(f"line {line}: unterminated string")
            toks.append(("id", "".join(buf), line))
            line += "".join(buf).count("\n")
            i = j + 1
        elif src.startswith("->", i):
            toks.append(("->", "->", line))
            i += 2
        elif c in "{}[]=,;":
            toks.append((c, c, line))
            i += 1
        else:
            m = re.match(r"[A-Za-z0-9_.$-]+", src[i:])
            if not m:
                raise RegistryError(f"line {line}: unexpected character {c!r}")
            toks.append(("id", m.group(0), line))
            i += len(m.group(0))
    return toks


def parse_dot(src):
    toks, pos = tokenize(src), 0

    def peek(k=0):
        return toks[pos + k] if pos + k < len(toks) else ("eof", "", -1)

    def take(kind):
        nonlocal pos
        t = peek()
        if t[0] != kind:
            raise RegistryError(f"line {t[2]}: expected {kind}, found {t[1]!r}")
        pos += 1
        return t[1]

    def attrs():
        out = {}
        if peek()[0] != "[":
            return out
        take("[")
        while peek()[0] != "]":
            k = take("id")
            take("=")
            out[k] = take("id")
            if peek()[0] in ",;":
                take(peek()[0])
        take("]")
        return out

    if take("id") != "digraph":
        raise RegistryError("registry must be a digraph")
    graph = {"name": take("id"), "attrs": {}, "nodes": {}, "edges": []}
    take("{")
    while peek()[0] != "}":
        ident = take("id")
        if ident in ("graph", "node", "edge") and peek()[0] == "[":
            if ident == "graph":
                graph["attrs"].update(attrs())
            else:
                attrs()
        elif peek()[0] == "=":
            take("=")
            graph["attrs"][ident] = take("id")
        else:
            chain = [ident]
            while peek()[0] == "->":
                take("->")
                chain.append(take("id"))
            a = attrs()
            if len(chain) == 1:
                graph["nodes"].setdefault(ident, {}).update(a)
            else:
                for n in chain:
                    graph["nodes"].setdefault(n, {})
                for frm, to in zip(chain, chain[1:]):
                    graph["edges"].append({"from": frm, "to": to, **a})
        if peek()[0] == ";":
            take(";")
    take("}")
    return graph


# --- skill sources ---------------------------------------------------------------


def frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep:
            out[k.strip()] = v.strip()
    return out


def slug(heading):
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return s.replace(" ", "-")


def anchors(text):
    return {slug(h) for h in re.findall(r"^#{1,6} (.+)$", text, re.M)}


def next_items(text):
    m = re.search(r"^## Next\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return None
    return [NEXT_ITEM.match(l).groups() for l in m.group(1).splitlines() if NEXT_ITEM.match(l)]


def load_skills():
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        if path.parent.name.startswith("_"):
            continue
        text = path.read_text()
        skills[path.parent.name] = {"text": text, "meta": frontmatter(text)}
    return skills


# --- validation ------------------------------------------------------------------


def team_specs(graph):
    specs = {}
    for k, v in graph["attrs"].items():
        if k.startswith(TEAM_KEY):
            specs[k[len(TEAM_KEY):]] = [s.strip() for s in v.split(">")]
    return specs


def validate(graph, skills):
    errs = []
    nodes, edges = graph["nodes"], graph["edges"]
    for sid, s in skills.items():
        if not SKILL_ID.match(sid):
            errs.append(f"skills/{sid}: id is not lowercase-hyphenated")
        if s["meta"].get("name") != sid:
            errs.append(f"skills/{sid}/SKILL.md: frontmatter name is {s['meta'].get('name')!r}")
        if not s["meta"].get("description"):
            errs.append(f"skills/{sid}/SKILL.md: missing description")
    for nid, a in nodes.items():
        t = a.get("type")
        if t not in NODE_TYPES:
            errs.append(f"node {nid}: type={t!r} is not one of {', '.join(NODE_TYPES)}")
        elif t == "skill" and nid not in skills:
            errs.append(f"node {nid}: type=skill but skills/{nid}/SKILL.md does not exist")
        elif t != "skill" and nid in skills:
            errs.append(f"node {nid}: skills/{nid}/ exists but type={t}")
    for e in edges:
        where = f"edge {e['from']} -> {e['to']}"
        if e.get("rel") not in RELS:
            errs.append(f"{where}: rel={e.get('rel')!r} is not one of {', '.join(RELS)}")
        if e.get("origin") not in ORIGINS:
            errs.append(f"{where}: origin={e.get('origin')!r} is not one of {', '.join(ORIGINS)}")
        if not e.get("workflows"):
            errs.append(f"{where}: no workflows=")
        src = source_of(e, nodes)
        if src is None:
            errs.append(f"{where}: from a non-skill node, so it needs source=")
            continue
        file, _, anchor = src.partition("#")
        path = ROOT / file
        if not path.is_file():
            errs.append(f"{where}: source {file} does not exist")
        elif anchor and anchor not in anchors(path.read_text()):
            errs.append(f"{where}: source {src} has no #{anchor} heading")
    for sid, s in skills.items():
        stated = next_items(s["text"])
        registered = {(e["to"], e.get("rel")) for e in edges if e["from"] == sid}
        if stated is None:
            if registered:
                errs.append(f"skills/{sid}/SKILL.md: has registry edges but no ## Next section")
            continue
        for item in set(stated) - registered:
            errs.append(f"skills/{sid}/SKILL.md: Next lists `{item[0]}` ({item[1]}), not in the registry")
        for item in registered - set(stated):
            errs.append(f"registry: {sid} -> {item[0]} ({item[1]}) is missing from skills/{sid}/SKILL.md Next")
    for wf, steps in team_specs(graph).items():
        for sid in steps:
            if nodes.get(sid, {}).get("type") != "skill":
                errs.append(f"team:{wf}: {sid} is not a skill node")
        for frm, to in zip(steps, steps[1:]):
            if not any(
                e["from"] == frm and e["to"] == to and wf in workflows(e) and e.get("rel") in ("transition", "suggested")
                for e in edges
            ):
                errs.append(f"team:{wf}: no transition {frm} -> {to} in workflow {wf}")
        if len(steps) < 2:
            errs.append(f"team:{wf}: needs a worker and at least one reviewer")
    return errs


def source_of(edge, nodes):
    if edge.get("source"):
        return edge["source"]
    if nodes.get(edge["from"], {}).get("type") == "skill":
        return f"skills/{edge['from']}/SKILL.md#next"
    return None


def workflows(edge):
    return [w.strip() for w in edge.get("workflows", "").split(",") if w.strip()]


# --- rendering -------------------------------------------------------------------


def mid(nid):
    return re.sub(r"\W", "_", nid)


def mermaid(graph, edges):
    nodes = graph["nodes"]
    used = []
    for e in edges:
        for n in (e["from"], e["to"]):
            if n not in used:
                used.append(n)
    out = ["```mermaid", "flowchart LR"]
    for n in used:
        a = nodes[n]
        label = a.get("label", n).replace('"', "'")
        shape = {"external": '(["{}"])', "action": '{{{{"{}"}}}}'}.get(a.get("type"), '["{}"]')
        out.append(f"  {mid(n)}{shape.format(label)}")
    for e in edges:
        label = e.get("artifact", "")
        if e["rel"] == "alternative":
            label = f"alternative: {label}"
        out.append(f'  {mid(e["from"])} {ARROWS[e["rel"]]}|"{label}"| {mid(e["to"])}')
    ext = [mid(n) for n in used if nodes[n].get("type") == "external"]
    act = [mid(n) for n in used if nodes[n].get("type") == "action"]
    out.append("  classDef external stroke-dasharray: 4 3")
    out.append("  classDef action fill:#fff4d6,stroke:#b8860b")
    if ext:
        out.append(f"  class {','.join(ext)} external")
    if act:
        out.append(f"  class {','.join(act)} action")
    out.append("```")
    return "\n".join(out)


def workflow_order(edges):
    order = []
    for e in edges:
        for w in workflows(e):
            if w not in order:
                order.append(w)
    return order


def unconnected(graph, skills):
    linked = {n for e in graph["edges"] for n in (e["from"], e["to"])}
    return [s for s in skills if s not in linked]


LEGEND = """| Arrow | Relationship |
| --- | --- |
| `==>` solid, thick | prerequisite: the target needs the source's output first |
| `-->` solid | transition stated in the source skill's Next section |
| `-.->` dashed | suggested next step; it does not authorize running it |
| `--o` circle end, label starts "alternative:" | alternative to the source for a case it names |

Node shapes: rectangle = skill in this repo; rounded, dashed outline = external skill or tool;
hexagon = non-skill action (a person or host acts). Edge labels name the artifact passed."""


def render_chains(graph, skills):
    edges, nodes = graph["edges"], graph["nodes"]
    teams = team_specs(graph)
    parts = [
        "# Skill chains",
        "",
        "<!-- Generated by scripts/chains.py from chains/skills.dot. Do not edit. -->",
        "",
        "How the skills in this repo hand work to each other and to external skills.",
        "Regenerate with `python3 scripts/chains.py`; verify with `python3 scripts/chains.py --check`.",
        "",
        "## Legend",
        "",
        LEGEND,
        "",
        "## Full graph",
        "",
        mermaid(graph, edges),
        "",
        "## Workflows",
    ]
    for wf in workflow_order(edges):
        parts += ["", f"### {wf}", ""]
        if wf in teams:
            parts += [
                f"Runnable with wteam: `wteam run --team <path to this repo>/chains/teams/{wf}.dot`,"
                f" or install it as a saved team (see README) and use `--team {wf}`"
                f" (worker `{teams[wf][0]}`, reviewers {', '.join(f'`{s}`' for s in teams[wf][1:])}).",
                "",
            ]
        parts.append(mermaid(graph, [e for e in edges if wf in workflows(e)]))
    parts += [
        "",
        "## Relationships",
        "",
        "Origin: observed = seen in the work artifacts this library was built from;"
        " inferred = added during design.",
        "",
        "| From | To | Relationship | Origin | Artifact | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for e in edges:
        src = source_of(e, nodes)
        parts.append(
            f"| `{e['from']}` | `{e['to']}` | {e['rel']} | {e['origin']} | {e.get('artifact', '')} | [{src}]({src}) |"
        )
    others = [(n, a) for n, a in nodes.items() if a.get("type") != "skill"]
    parts += ["", "## Not skills in this repo", "", "| Node | Type | What |", "| --- | --- | --- |"]
    parts += [f"| `{n}` | {a['type']} | {a.get('label', n)} |" for n, a in others]
    parts += ["", "## Unconnected skills", ""]
    parts += [f"- `{s}`" for s in unconnected(graph, skills)] or ["None."]
    return "\n".join(parts) + "\n"


def render_readme_section(graph, skills):
    lone = unconnected(graph, skills)
    return "\n".join(
        [
            README_START,
            "<!-- Generated by scripts/chains.py. Do not edit. -->",
            "",
            mermaid(graph, graph["edges"]),
            "",
            "Solid arrows are transitions a skill states, dashed arrows are suggestions, and circle-ended"
            " arrows are alternatives. [CHAINS.md](CHAINS.md) has the legend, one diagram per workflow,"
            " the runnable wteam teams, and where each relationship comes from.",
            "",
            f"Skills with no chain: {', '.join(f'`{s}`' for s in lone)}." if lone else "Every skill is chained.",
            README_END,
        ]
    )


def dot_str(s):
    return '"' + s.replace('"', '\\"') + '"'


def first_sentence(text):
    m = re.match(r"(.+?\.)(\s|$)", text)
    return m.group(1) if m else text


def render_team(wf, steps, skills):
    worker, reviewers = steps[0], steps[1:]
    description = (
        f"{first_sentence(skills[worker]['meta']['description'])}"
        f" Worker follows {worker}; reviewed with {', '.join(reviewers)}."
    )
    roles = ["worker", "reviewer"] + list(reviewers[1:])

    def prompt(sid, part):
        return dot_str(f"Follow the {sid} skill ({part}). If it is not installed, read skills/{sid}/SKILL.md in bcmyguest/personal-skills.")

    lines = [
        f"// Generated by scripts/chains.py from chains/skills.dot (team:{wf}). Do not edit.",
        f"// {' -> '.join(steps)}. Landing (push, PR) stays outside the run: see land-change.",
        f"digraph {mid(wf)} {{",
        f"  graph [description={dot_str(description)}, max_visits=3, handoff_tokens=150000]",
        '  planner      [agent="planner"]',
        '  orchestrator [agent="orchestrator"]',
        f'  worker       [agent="worker", count="$pairs", label={dot_str(worker)}, skill={dot_str(worker)}, prompt={prompt(worker, "worker")}]',
    ]
    for role, sid in zip(roles[1:], reviewers):
        lines.append(f'  {dot_str(role)} [agent="reviewer", count="$pairs", label={dot_str(sid)}, skill={dot_str(sid)}, prompt={prompt(sid, "reviewer")}]')
    lines.append("  orchestrator -> worker [on=dispatch, kind=broadcast]")
    lines.append(f"  worker -> {dot_str(roles[1])} [on=submit, kind=pair]")
    for i, role in enumerate(roles[1:], start=1):
        lines.append(f"  {dot_str(role)} -> worker [on=fail, kind=pair]")
        nxt = roles[i + 1] if i + 1 < len(roles) else None
        if nxt:
            lines.append(f"  {dot_str(role)} -> {dot_str(nxt)} [on=pass, kind=pair]")
        else:
            lines.append(f"  {dot_str(role)} -> orchestrator [on=pass, kind=fan-in]")
    lines.append("  worker -> orchestrator [on=blocked, kind=fan-in]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def splice_readme(text, section):
    if README_START in text and README_END in text:
        head, _, rest = text.partition(README_START)
        _, _, tail = rest.partition(README_END)
        return head + section + tail
    raise RegistryError(f"README.md has no {README_START} ... {README_END} markers")


def outputs(graph, skills):
    files = {CHAINS_MD: render_chains(graph, skills)}
    files[README] = splice_readme(README.read_text(), render_readme_section(graph, skills))
    for wf, steps in team_specs(graph).items():
        files[TEAMS_DIR / f"{wf}.dot"] = render_team(wf, steps, skills)
    return files


def run_wteam(paths):
    wteam = shutil.which("wteam")
    if not wteam:
        print("wteam not installed: skipped the wteam-level check of the DOT files")
        return []
    errs = []
    for p in paths:
        r = subprocess.run([wteam, "graph", "--team", str(p)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            errs.append(f"wteam rejects {p.relative_to(ROOT)}: {(r.stderr or r.stdout).strip()}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail when outputs are stale or the registry is inconsistent")
    ap.add_argument("--wteam", action="store_true", help="also validate the DOT files with `wteam graph --team`")
    args = ap.parse_args()
    try:
        graph = parse_dot(REGISTRY.read_text())
        skills = load_skills()
        errs = validate(graph, skills)
        files = {} if errs else outputs(graph, skills)
    except RegistryError as e:
        errs, files = [str(e)], {}
    if not errs:
        expected = {p.name for p in files if p.parent == TEAMS_DIR}
        stray = [p for p in TEAMS_DIR.glob("*.dot") if p.name not in expected]
        if args.check:
            errs += [f"{p.relative_to(ROOT)} is stale" for p, text in files.items() if not p.is_file() or p.read_text() != text]
            errs += [f"{p.relative_to(ROOT)} has no team: entry in the registry" for p in stray]
        else:
            TEAMS_DIR.mkdir(parents=True, exist_ok=True)
            for p, text in files.items():
                if not p.is_file() or p.read_text() != text:
                    p.write_text(text)
                    print(f"wrote {p.relative_to(ROOT)}")
            for p in stray:
                p.unlink()
                print(f"removed {p.relative_to(ROOT)}")
    if not errs and args.wteam:
        errs += run_wteam([REGISTRY, *sorted(TEAMS_DIR.glob("*.dot"))])
    for e in errs:
        print(f"error: {e}", file=sys.stderr)
    if errs:
        sys.exit(1)
    if args.check:
        print("chains: up to date")


if __name__ == "__main__":
    main()
