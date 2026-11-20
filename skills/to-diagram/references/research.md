# Research behind to-diagram

Gathered November 2026. Each rule in `SKILL.md` traces to a finding here.

## When a diagram beats prose

- Larkin and Simon (1987), *Why a Diagram is (Sometimes) Worth Ten Thousand Words*, Cognitive Science 11:65–99 ([PDF](https://mechanism.ucsd.edu/bill/teaching/F12/cs200/Readings/larkin.whyadiagramissometimesworth.1987.pdf)). Diagrams index information by location, so related facts sit next to each other and the reader searches less; they also make explicit what text leaves implicit. The gain is in search and inference, so it appears when the reader must relate many parts. Basis for step 1.
- [Mermaid as agent output format](https://agentpatterns.ai/instructions/mermaid-as-agent-output-format/) (agentpatterns.ai, marked "emerging"). Use Mermaid for graph-shaped content only; linear procedures, tabular comparisons, and numbers read faster as a list or table. The choice is gated by the surface, not the model. Basis for steps 1 and 3.
- Financial Times, [Visual Vocabulary](https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/Visual-vocabulary.pdf). Picks a chart by the relationship to show: deviation, correlation, ranking, distribution, change over time, magnitude, part-to-whole, spatial, flow. Basis for the chart line in step 1.

## Picking the type

- Simon Brown, [C4 model](https://c4model.com/): one diagram per zoom level (context, container, component, code). Sequence, state, and ER diagrams fall outside C4 and complement it. Basis for the boxes row and the one-zoom-level rule.
- UML practice: sequence diagrams for time-ordered messages, state machines for lifecycles, class diagrams for type structure. The same mapping appears in the agentpatterns page and in published diagram-choice skills (`diagram-patterns`, `cathrynlavery/diagram-design`).

## Where diagrams render

| Surface | Mermaid | Source |
| --- | --- | --- |
| GitHub, GitLab, Azure DevOps, Gitea | Yes | [Mermaid integrations](https://mermaid.js.org/ecosystem/integrations-community.html) |
| VS Code Markdown preview, notebooks, chat | Yes, from 1.121 (May 2026) | [VS Code 1.121 notes](https://code.visualstudio.com/updates/v1_121) |
| Cursor CLI | ASCII rendering, from Feb 2026 | [Cursor changelog](https://cursor.com/changelog/cli-feb-18-2026) |
| Kiro CLI | Yes, from 2.31.1 (Oct 2026) | [Kiro changelog](https://kiro.dev/changelog/cli/2-31-1/) |
| Claude Code terminal | No; shows source | [claude-code#14375](https://github.com/anthropics/claude-code/issues/14375) (open) |
| Claude Desktop, Claude Code tab | No; claude.ai web does | [claude-code#52517](https://github.com/anthropics/claude-code/issues/52517) |
| Plain terminal, email, Slack | No | agentpatterns.ai |

Terminal fallbacks exist as renderers: `mermaid-ascii` (Go), `beautiful-mermaid`, and the Rust renderer in Grok CLI ([Simon Willison, July 2026](https://simonwillison.net/2026/Jul/16/grok-mermaid/)). Basis for the terminal row of step 3.

## Size and correctness

- Mermaid Chart: flowchart layout degrades past about 100 connections ([blog](https://docs.mermaidchart.com/blog/posts/flow-charts-are-on2-complex-so-dont-go-over-100-connections)); Mermaid-Sonar flags 50 nodes as complex. The 20-node cap follows the agentpatterns example prompt and keeps a diagram within one screen.
- [MermaidSeqBench](https://arxiv.org/abs/2511.14967) (IBM Research, NeurIPS 2025 workshop): state-of-the-art models show significant gaps in producing correct Mermaid sequence diagrams. LLM-written Mermaid also fails silently on syntax, for example backticks in labels ([pr-agent#2211](https://github.com/The-PR-Agent/pr-agent/issues/2211)). Basis for the label and validation rules.

## Existing tools, and the gap this skill fills

| Tool | What it does | What it lacks |
| --- | --- | --- |
| Anthropic `eli5` plugin | Picture-heavy HTML explainer for a topic | Writes a separate page; does not pick a diagram type or target the chat |
| Cursor `/visualize` (Sep 2026) | Inline charts and diagrams in Cursor's Agents Window | Cursor only |
| Sourcebot "Visualize" button | Redraws the previous answer as Mermaid | Sourcebot only; Mermaid only |
| `beautiful-mermaid`, `claude-mermaid-ascii` | Render Mermaid as terminal art | Rendering only; no choice of type or of whether to diagram |
| `diagram-patterns`, `mermaid-diagrams` skills | Type decision tree, Mermaid syntax | No surface check; write to files rather than the chat |

`to-diagram` combines the three decisions the others split: whether to diagram, which type, and which format renders here.
