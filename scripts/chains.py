#!/usr/bin/env python3
"""Check and draw skill chains from wteam/teams/*/team.dot and each skill's ## Next section.

Writes CHAINS.md and the README "Skill chains" section. --check fails on stale output, a
team step whose skill is missing, or a team hand-off no ## Next section states; --wteam also
runs `wteam graph --team` on every team file when wteam is installed.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEAMS_DIR = ROOT / "wteam" / "teams"
SKILLS_DIR = ROOT / "skills"
CHAINS_MD = ROOT / "CHAINS.md"
README = ROOT / "README.md"
README_START = "<!-- chains:start -->"
README_END = "<!-- chains:end -->"

RELS = ("prerequisite", "transition", "suggested", "alternative")
SKILL_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NEXT_ITEM = re.compile(r"^- `([a-z0-9-]+)` \(([a-z]+)\): (.+)$")
ARROWS = {
    "prerequisite": "==>",
    "transition": "-->",
    "suggested": "-.->",
    "alternative": "--o",
}
# Next targets that are not skills in this repo: (type, label). type is external or action.
EXTERNAL = {
    "wteam": ("external", "wteam (bcmyguest/wteam)"),
    "refine": ("external", "refine (bcmyguest/wteam)"),
    "bugbot": ("external", "Bugbot"),
    "to-spec": ("external", "to-spec (mattpocock/skills)"),
    "spend-approval": ("action", "user approves spend"),
    "next-agent": ("action", "start next agent"),
}
CONTROL_ROLES = ("planner", "orchestrator")


class ChainError(Exception):
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
                raise ChainError(f"line {line}: unterminated /* comment")
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
                raise ChainError(f"line {line}: unterminated string")
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
                raise ChainError(f"line {line}: unexpected character {c!r}")
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
            raise ChainError(f"line {t[2]}: expected {kind}, found {t[1]!r}")
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
        raise ChainError("a team must be a digraph")
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


# --- sources ---------------------------------------------------------------------


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


def next_items(text):
    """`(target, rel, artifact)` per ## Next bullet; None when the skill has no ## Next."""
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
        skills[path.parent.name] = {"text": text, "meta": frontmatter(text), "next": next_items(text) or []}
    return skills


def load_teams():
    teams = {}
    for path in sorted(TEAMS_DIR.glob("*/team.dot")):
        try:
            teams[path.parent.name] = parse_dot(path.read_text())
        except ChainError as e:
            raise ChainError(f"{path.relative_to(ROOT)}: {e}") from None
    return teams


def handoffs(skills):
    return [
        {"from": sid, "to": to, "rel": rel, "artifact": artifact}
        for sid, s in skills.items()
        for to, rel, artifact in s["next"]
    ]


# --- teams -----------------------------------------------------------------------


def kind(node):
    shape = node.get("shape", "box")
    return {"box": "agent", "hexagon": "human", "parallelogram": "command"}.get(shape, shape)


def routes(graph, frm, on):
    return [e["to"] for e in graph["edges"] if e["from"] == frm and e.get("on") == on]


def is_step(graph, nid):
    return nid not in CONTROL_ROLES and kind(graph["nodes"][nid]) == "agent" and bool(routes(graph, nid, "submit"))


def flow(graph):
    """Nodes a ticket visits in order: the dispatch target, then each submit or pass hop."""
    start = routes(graph, "orchestrator", "dispatch")
    if not start:
        return []
    order, at = [start[0]], start[0]
    while True:
        nxt = [t for t in routes(graph, at, "submit") + routes(graph, at, "pass") if t not in order and t not in CONTROL_ROLES]
        if not nxt:
            return order
        at = nxt[0]
        order.append(at)


def validate(teams, skills):
    errs = []
    for sid, s in skills.items():
        if not SKILL_ID.match(sid):
            errs.append(f"skills/{sid}: id is not lowercase-hyphenated")
        if s["meta"].get("name") != sid:
            errs.append(f"skills/{sid}/SKILL.md: frontmatter name is {s['meta'].get('name')!r}")
        if not s["meta"].get("description"):
            errs.append(f"skills/{sid}/SKILL.md: missing description")
        for to, rel, _ in s["next"]:
            if rel not in RELS:
                errs.append(f"skills/{sid}/SKILL.md: Next `{to}` has ({rel}), not one of {', '.join(RELS)}")
            if to not in skills and to not in EXTERNAL:
                errs.append(f"skills/{sid}/SKILL.md: Next `{to}` is neither a skill here nor a known external")
    for name, g in teams.items():
        where = f"wteam/teams/{name}/team.dot"
        if g["name"] != name.replace("-", "_"):
            errs.append(f"{where}: digraph is named {g['name']}, expected {name.replace('-', '_')}")
        if not g["attrs"].get("description"):
            errs.append(f"{where}: no graph description")
        path = flow(g)
        if not path:
            errs.append(f"{where}: the orchestrator dispatches to no node")
            continue
        for nid, a in g["nodes"].items():
            if nid in CONTROL_ROLES or kind(a) != "agent":
                continue
            if a.get("skill") not in skills:
                errs.append(f"{where}: node {nid} has skill={a.get('skill')!r}, not a skill in skills/")
        step = None
        agents = [n for n in path if kind(g["nodes"][n]) == "agent"]
        for x, y in zip(agents, agents[1:]):
            step = x if is_step(g, x) else step
            sx, sy = g["nodes"][x].get("skill"), g["nodes"][y].get("skill")
            if sx not in skills or sy not in skills or sx == sy:
                continue
            from_step = g["nodes"][step].get("skill") if step else sx
            stated = {t for t, _, _ in skills[sx]["next"]} | {t for t, _, _ in skills.get(from_step, {"next": []})["next"]}
            if sy not in stated:
                errs.append(
                    f"{where}: {x} ({sx}) hands to {y} ({sy}), but no ## Next in skills/{sx}"
                    + (f" or skills/{from_step}" if from_step != sx else "")
                    + f" lists `{sy}`"
                )
    return errs


# --- rendering -------------------------------------------------------------------


def mid(nid):
    return re.sub(r"\W", "_", nid)


def short(artifact):
    s = re.split(r"[,.;]", artifact)[0].strip()
    return re.sub(r"^(the|a|an) ", "", s).replace('"', "'")


def handoff_mermaid(edges, skills):
    used = []
    for e in edges:
        for n in (e["from"], e["to"]):
            if n not in used:
                used.append(n)
    out = ["```mermaid", "flowchart LR"]
    for n in used:
        t, label = EXTERNAL.get(n, ("skill", n))
        shape = {"external": '(["{}"])', "action": '{{{{"{}"}}}}'}.get(t, '["{}"]')
        out.append(f"  {mid(n)}{shape.format(label)}")
    for e in edges:
        label = short(e["artifact"])
        if e["rel"] == "alternative":
            label = f"alternative: {label}"
        out.append(f'  {mid(e["from"])} {ARROWS[e["rel"]]}|"{label}"| {mid(e["to"])}')
    ext = [mid(n) for n in used if EXTERNAL.get(n, ("",))[0] == "external"]
    act = [mid(n) for n in used if EXTERNAL.get(n, ("",))[0] == "action"]
    out.append("  classDef external stroke-dasharray: 4 3")
    out.append("  classDef action fill:#fff4d6,stroke:#b8860b")
    if ext:
        out.append(f"  class {','.join(ext)} external")
    if act:
        out.append(f"  class {','.join(act)} action")
    out.append("```")
    return "\n".join(out)


def team_mermaid(g):
    path = flow(g)
    out = ["```mermaid", "flowchart LR"]
    for n in path:
        a = g["nodes"][n]
        k = kind(a)
        if k == "human":
            out.append(f'  {mid(n)}{{{{"{a.get("label", n)}"}}}}')
        elif k == "command":
            out.append(f'  {mid(n)}[/"{a.get("script", n)}"/]')
        else:
            out.append(f'  {mid(n)}["{n}: {a.get("skill", "")}"]')
    out.append('  done(["commit"])')
    for x, y in zip(path, path[1:] + ["done"]):
        on = "submit" if y in routes(g, x, "submit") else "pass"
        out.append(f"  {mid(x)} -->|{on}| {mid(y)}")
    for e in g["edges"]:
        if e.get("on") == "fail":
            out.append(f"  {mid(e['from'])} -.->|fail| {mid(e['to'])}")
    steps = [mid(n) for n in path if is_step(g, n)]
    out.append("  classDef step fill:#e8f0fe,stroke:#4a6fa5")
    if steps:
        out.append(f"  class {','.join(steps)} step")
    out.append("```")
    return "\n".join(out)


def unconnected(teams, skills):
    linked = {n for e in handoffs(skills) for n in (e["from"], e["to"])}
    linked |= {a.get("skill") for g in teams.values() for a in g["nodes"].values()}
    return [s for s in skills if s not in linked]


LEGEND = """| Arrow | Relationship |
| --- | --- |
| `==>` solid, thick | prerequisite: the target needs the source's output first |
| `-->` solid | transition stated in the source skill's Next section |
| `-.->` dashed | suggested next step; it does not authorize running it |
| `--o` circle end, label starts "alternative:" | alternative to the source for a case it names |

Node shapes: rectangle = skill in this repo; rounded, dashed outline = external skill or tool;
hexagon = non-skill action (a person or host acts). Edge labels name the artifact passed."""


def steps_text(g):
    parts = []
    for n in flow(g):
        a = g["nodes"][n]
        if kind(a) == "human":
            parts.append(f"gate *{a.get('label', n)}*")
        elif kind(a) == "agent":
            parts.append(f"`{a.get('skill')}`" + ("" if is_step(g, n) else " (review)"))
    return " → ".join(parts)


def render_chains(teams, skills):
    edges = handoffs(skills)
    parts = [
        "# Skill chains",
        "",
        "<!-- Generated by scripts/chains.py from wteam/teams/ and each skill's ## Next. Do not edit. -->",
        "",
        "Teams are runnable [wteam](https://github.com/bcmyguest/wteam) teams in `wteam/teams/`.",
        "Install them with `scripts/install-wteam.sh`, then `wteam run --pairs N --team <name>`.",
        "A ticket walks every step of its team and commits once, after the last review passes.",
        "Hand-offs are what each skill's `## Next` section states, inside or outside a team.",
        "Regenerate with `python3 scripts/chains.py`; verify with `python3 scripts/chains.py --check`.",
        "",
        "## Teams",
    ]
    for name, g in teams.items():
        parts += ["", f"### {name}", "", g["attrs"].get("description", ""), "", f"Path: {steps_text(g)}.", "", team_mermaid(g)]
    parts += [
        "",
        "## Hand-offs",
        "",
        LEGEND,
        "",
        handoff_mermaid(edges, skills),
        "",
        "| From | To | Relationship | Artifact |",
        "| --- | --- | --- | --- |",
    ]
    parts += [f"| `{e['from']}` | `{e['to']}` | {e['rel']} | {e['artifact']} |" for e in edges]
    parts += ["", "## Not skills in this repo", "", "| Node | Type | What |", "| --- | --- | --- |"]
    parts += [f"| `{n}` | {t} | {label} |" for n, (t, label) in EXTERNAL.items()]
    parts += ["", "## Unconnected skills", ""]
    parts += [f"- `{s}`" for s in unconnected(teams, skills)] or ["None."]
    return "\n".join(parts) + "\n"


def render_readme_section(teams, skills):
    lone = unconnected(teams, skills)
    rows = [f"| `{name}` | {steps_text(g)} |" for name, g in teams.items()]
    return "\n".join(
        [
            README_START,
            "<!-- Generated by scripts/chains.py. Do not edit. -->",
            "",
            "| Team | Path |",
            "| --- | --- |",
            *rows,
            "",
            handoff_mermaid(handoffs(skills), skills),
            "",
            "Solid arrows are transitions a skill states, dashed arrows are suggestions, and circle-ended"
            " arrows are alternatives. [CHAINS.md](CHAINS.md) draws each team and lists every hand-off.",
            "",
            f"Skills with no chain: {', '.join(f'`{s}`' for s in lone)}." if lone else "Every skill is chained.",
            README_END,
        ]
    )


def splice_readme(text, section):
    if README_START in text and README_END in text:
        head, _, rest = text.partition(README_START)
        _, _, tail = rest.partition(README_END)
        return head + section + tail
    raise ChainError(f"README.md has no {README_START} ... {README_END} markers")


def outputs(teams, skills):
    return {
        CHAINS_MD: render_chains(teams, skills),
        README: splice_readme(README.read_text(), render_readme_section(teams, skills)),
    }


def run_wteam(paths):
    wteam = shutil.which("wteam")
    if not wteam:
        print("wteam not installed: skipped the wteam-level check of the team files")
        return []
    errs = []
    for p in paths:
        r = subprocess.run([wteam, "graph", "--team", str(p)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            errs.append(f"wteam rejects {p.relative_to(ROOT)}: {(r.stderr or r.stdout).strip()}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail when outputs are stale or a team disagrees with the skills")
    ap.add_argument("--wteam", action="store_true", help="also validate the team files with `wteam graph --team`")
    args = ap.parse_args()
    try:
        teams, skills = load_teams(), load_skills()
        errs = validate(teams, skills)
        files = {} if errs else outputs(teams, skills)
    except ChainError as e:
        errs, files = [str(e)], {}
    if not errs:
        if args.check:
            errs += [f"{p.relative_to(ROOT)} is stale" for p, text in files.items() if not p.is_file() or p.read_text() != text]
        else:
            for p, text in files.items():
                if not p.is_file() or p.read_text() != text:
                    p.write_text(text)
                    print(f"wrote {p.relative_to(ROOT)}")
    if not errs and args.wteam:
        errs += run_wteam(sorted(TEAMS_DIR.glob("*/team.dot")))
    for e in errs:
        print(f"error: {e}", file=sys.stderr)
    if errs:
        sys.exit(1)
    if args.check:
        print("chains: up to date")


if __name__ == "__main__":
    main()
