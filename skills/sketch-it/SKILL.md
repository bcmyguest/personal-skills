---
name: sketch-it
description: Turn the current topic into the view a person can read fastest, a text sketch, a diagram, or one HTML page. Use when the user runs /sketch-it or asks for a sketch, a diagram, a page, or to see the shape or the flow.
argument-hint: "[item numbers, comma-separated] [what to show]"
---

# Sketch it

Draw the subject as one view the user can read faster than the prose. The reasons behind each rule are in [references/research.md](references/research.md). Shapes for the text views are in [references/sketches.md](references/sketches.md).

## Arguments

The text after the command (`$ARGUMENTS` in hosts that substitute it) holds up to two parts, both optional:

- **Item numbers,** comma-separated. They refer to the numbered items in your last response; with no numbering, count its top-level bullets or paragraphs from 1.
- **Subject,** any text after the numbers ("the auth flow", "how the queue retries").

No arguments: show the last response as the easiest view.

## 1. Pick the view the reader can take in

A view pays when the reader must relate many parts at once. Pick the row that takes the least searching. One view; add a second only when it carries a distinct second point.

A tree, list, or table in the chat is easier when indentation or a column already indexes the facts. One HTML page is easier when the reader must see where something sits, compare two appearances, or read labels placed on a picture. A diagram is easier when the links are the point. A page with navigation is harder than one file: it adds searching.

| The point is | View | Then |
| --- | --- | --- |
| A linear procedure | A numbered list | Step 5 |
| Options on shared criteria | A table | Step 5 |
| Branching logic or an algorithm | Pseudocode | Step 5 |
| Who calls whom, and nesting matters more than message order | Call tree | Step 5 |
| Which files own what | File tree, one or two levels | Step 5 |
| UI ownership, the state and the module boundaries, not where pixels sit | Component tree | Step 5 |
| What changes in a tree or in branching logic the reader already has | A diff of that shape | Step 5 |
| A new block to copy, or omitted lines would hide ownership or order | The whole block, read from the file now | Step 5 |
| Where regions sit, two appearances compared, or labels that have to sit on the picture | One HTML page from a JSON spec, via `node <this skill's folder>/scripts/page.mjs SPEC -o ${TMPDIR:-/tmp}/sketch-it/<slug>.html` | Step 5. The command prints the path. Open it on this machine (`xdg-open` on Linux, `open` on macOS). When you run on another machine, send the file with the host's file-sharing tool |
| Numbers, or links a tree cannot show (message order, cycles, state transitions, cross-links, typed relations) | A diagram | Step 2 |

Read the matching section of [references/sketches.md](references/sketches.md) before writing a sketch, a shape diff, a whole block, or the HTML page.

## 2. Pick the diagram type by the question it answers

| Question | Type | Mermaid |
| --- | --- | --- |
| Who sends what to whom, in what order? | Sequence | `sequenceDiagram` |
| What states can it be in, and what moves it between them? | State machine | `stateDiagram-v2` |
| Which path does control take, and where does it branch? | Sequence with branches | `sequenceDiagram` with `alt`, `opt`, `loop` |
| What are the parts and how do they connect? | Sequence of the calls between parts, one zoom level (context, container, or component) | `sequenceDiagram`, participants ordered by boundary |
| What data exists and how does it relate? | Entity-relationship | `erDiagram` |
| What types exist and how do they extend or hold each other? | Class | `classDiagram` |
| What changes in the links between existing parts? | The same type, changed nodes marked | `classDef changed` |
| What happens when? | Timeline | `gantt` or a table |
| How do numbers relate? | Chart: change over time is a line; ranking is sorted bars; part-to-whole is a stacked bar; distribution is a histogram; correlation is a scatter | `xychart-beta` or `pie` |

When two questions apply, draw two small diagrams rather than one mixed one.

Use `flowchart` only when the user asks for one. A sequence diagram reads faster: it shows who acts, in what order, top to bottom.

## 3. Pick the format by where it will render

| Surface | Format |
| --- | --- |
| Host has an inline visual tool (for example a widget or artifact tool) | Use that tool. |
| Renders Mermaid inline: GitHub, GitLab, Azure DevOps, claude.ai, VS Code 1.121+ chat and preview, Cursor IDE chat, Zed, Obsidian | A fenced `mermaid` block. |
| Anything else: terminal agents (Claude Code CLI, Codex CLI, opencode), desktop app tabs that show raw source, unknown hosts | A file. Write the Mermaid to `${TMPDIR:-/tmp}/sketch-it/<slug>.mmd`, run `node <this skill's folder>/scripts/diagram.mjs html <that file>`, and open the printed page the same way as any other HTML page in step 1. |

The HTML page is self-contained (Mermaid is inlined), so it opens offline. When the agent runs on a different machine from the user, send the file with the host's file-sharing tool instead of a path.

## 4. Draw

1. At most 20 nodes per diagram. Past that, split by zoom level: one overview, then one diagram per part.
2. Every node and edge maps to something real. For code, name the file in the node or list `path:line` under the diagram. Invent no components.
3. Label edges with the verb or message ("validates", "POST /token"), not with "uses".
4. Labels hold no backticks; quote labels that contain punctuation (`A["parse(x)"]`).
5. Add a legend only when shape or colour carries meaning.
6. Check every diagram before sending: `node <this skill's folder>/scripts/diagram.mjs check FILE` runs Mermaid's own parser on a `.mmd` file or on each `mermaid` fence in a `.md` file, and fails on a syntax error or more than 20 nodes. Fix and rerun until it prints OK. A block that fails to parse renders as nothing on most surfaces. The first run installs `mermaid` and `jsdom` into the script's cache.

## 5. Explain

Deliver in this order: one line naming the point, the view, then at most two sentences — where to start reading, and the one thing the view shows that the prose hid.
