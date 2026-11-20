---
name: to-diagram
description: Turn what the conversation is about (the last response, chosen items from it, or a named topic) into the diagram that best shows it, drawn so it renders in the current chat when the host supports that. Use when the user runs /to-diagram or asks to see something as a diagram.
argument-hint: "[item numbers, comma-separated] [what to diagram]"
---

# To diagram

Draw the subject as one picture the user can read faster than the prose. The reasons behind each rule are in [references/research.md](references/research.md).

## Arguments

The text after the command (`$ARGUMENTS` in hosts that substitute it) holds up to two parts, both optional:

- **Item numbers,** comma-separated. They refer to the numbered items in your last response; with no numbering, count its top-level bullets or paragraphs from 1.
- **Subject,** any text after the numbers ("the auth flow", "how the queue retries").

No arguments: diagram the last response.

## 1. Check that a diagram fits

A diagram pays when the reader must relate many parts at once: who calls whom, what leads where, what contains what. Use something else when the content is:

- a linear procedure: a numbered list;
- a comparison of options on criteria: a table;
- numbers: a chart, picked by the relationship to show (change over time: line; ranking: sorted bars; part-to-whole: stacked bar; distribution: histogram; correlation: scatter).

Say which you chose instead, in one line, and produce it.

## 2. Pick the diagram type by the question it answers

| Question | Type | Mermaid |
| --- | --- | --- |
| Who sends what to whom, in what order? | Sequence | `sequenceDiagram` |
| What states can it be in, and what moves it between them? | State machine | `stateDiagram-v2` |
| Which path does control take, and where does it branch? | Flowchart | `flowchart TD` |
| What are the parts and how do they connect? | C4-style boxes, one zoom level (context, container, or component) | `flowchart LR` with `subgraph` per boundary |
| What data exists and how does it relate? | Entity-relationship | `erDiagram` |
| What types exist and how do they extend or hold each other? | Class | `classDiagram` |
| What changes in a proposed change? | Before and after, changed nodes marked | two `flowchart`s, or one with a `classDef changed` |
| What happens when? | Timeline | `gantt` or a table |

When two questions apply, draw two small diagrams rather than one mixed one.

## 3. Pick the format by where it will render

| Surface | Format |
| --- | --- |
| Host has an inline visual tool (for example a widget or artifact tool) | Use that tool. |
| Renders Mermaid inline: GitHub, GitLab, Azure DevOps, claude.ai, VS Code 1.121+ chat and preview, Cursor IDE chat, Zed, Obsidian | A fenced `mermaid` block. |
| Anything else: terminal agents (Claude Code CLI, Codex CLI, opencode), desktop app tabs that show raw source, unknown hosts | A file. Write the Mermaid to `${TMPDIR:-/tmp}/diagrams/<slug>.mmd`, run `node <this skill's folder>/scripts/diagram.mjs html <that file>`, and give the user the printed path to open in a browser. |

The HTML page is self-contained (Mermaid is inlined), so it opens offline. When the agent runs on a different machine from the user, send the file with the host's file-sharing tool instead of a path. Charts go through the same route: Mermaid's `xychart-beta` draws line and bar charts, and `pie` draws part-to-whole.

## 4. Draw

1. At most 20 nodes per diagram. Past that, split by zoom level: one overview, then one diagram per part.
2. Every node and edge maps to something real. For code, name the file in the node or list `path:line` under the diagram. Invent no components.
3. Label edges with the verb or message ("validates", "POST /token"), not with "uses".
4. Labels hold no backticks; quote labels that contain punctuation (`A["parse(x)"]`).
5. Add a legend only when shape or colour carries meaning.
6. Check every diagram before sending: `node <this skill's folder>/scripts/diagram.mjs check FILE` runs Mermaid's own parser on a `.mmd` file or on each `mermaid` fence in a `.md` file, and fails on a syntax error or more than 20 nodes. Fix and rerun until it prints OK. A block that fails to parse renders as nothing on most surfaces. The first run installs `mermaid` and `jsdom` into `~/.cache/to-diagram`.

## 5. Explain

After the diagram, at most three sentences: where to start reading, and the one thing the picture shows that the prose hid.
