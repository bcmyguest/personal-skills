# Research behind sketch-it

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

Terminal fallbacks exist as renderers: `mermaid-ascii` (Go), `beautiful-mermaid`, and the Rust renderer in Grok CLI ([Simon Willison, July 2026](https://simonwillison.net/2026/Jul/16/grok-mermaid/)). These draw an approximation of the layout. A page that runs Mermaid itself draws exactly what GitHub and VS Code draw, opens on any machine with a browser, and needs nothing on `PATH` but Node, so step 3 writes a file for every surface that cannot draw Mermaid.

## Size and correctness

- Mermaid Chart: flowchart layout degrades past about 100 connections ([blog](https://docs.mermaidchart.com/blog/posts/flow-charts-are-on2-complex-so-dont-go-over-100-connections)); Mermaid-Sonar flags 50 nodes as complex. The 20-node cap follows the agentpatterns example prompt and keeps a diagram within one screen.
- [MermaidSeqBench](https://arxiv.org/abs/2511.14967) (IBM Research, NeurIPS 2025 workshop): state-of-the-art models show significant gaps in producing correct Mermaid sequence diagrams. LLM-written Mermaid also fails silently on syntax, for example backticks in labels ([pr-agent#2211](https://github.com/The-PR-Agent/pr-agent/issues/2211)). Basis for the label rule and for `scripts/diagram.mjs check`, which runs Mermaid's own parser (`mermaid.parse`, headless under jsdom) so errors surface before the user sees a blank block.

## Existing tools, and the gap this skill fills

| Tool | What it does | What it lacks |
| --- | --- | --- |
| Anthropic `eli5` plugin | Picture-heavy HTML explainer for a topic | Writes a separate page; does not pick a diagram type or target the chat |
| Cursor `/visualize` (Sep 2026) | Inline charts and diagrams in Cursor's Agents Window | Cursor only |
| Sourcebot "Visualize" button | Redraws the previous answer as Mermaid | Sourcebot only; Mermaid only |
| `beautiful-mermaid`, `claude-mermaid-ascii` | Render Mermaid as terminal art | Rendering only; no choice of type or of whether to diagram |
| `diagram-patterns`, `mermaid-diagrams` skills | Type decision tree, Mermaid syntax | No surface check; write to files rather than the chat |
| [Humanlayer `show-me`](https://github.com/humanlayer/skills/blob/main/plugins/show-me/skills/show-me/SKILL.md) | A palette: pseudocode, call tree, component tree, file tree, a diff of that shape, a whole block, an HTML page, then Mermaid | No surface check, no parser, no size cap |

## Smaller than a diagram

Larkin and Simon's gain is search by location. A call tree, a file tree, and a component tree already place related facts in that way, and they render as text on every host. A graph layout earns its cost when the links are the point: message order, cycles, state transitions, cross-links, typed relations. [MermaidSeqBench](https://arxiv.org/abs/2511.14967) shows models often emit Mermaid that does not parse, so a tree drawn as a flowchart fails twice — it is the wrong picture, and it may not render.

[Humanlayer's `show-me`](https://github.com/humanlayer/skills/blob/main/plugins/show-me/skills/show-me/SKILL.md) enumerates that palette, including one HTML page for a layout, a state comparison, or a concept a diagram cannot place. Step 1 takes the palette and picks by which view the reader can take in. The shapes in `sketches.md` are ours.

## When a page is easier than a fence

- Mayer, *Multimedia Learning*, chapter 5, [spatial contiguity](https://www.cambridge.org/core/books/abs/multimedia-learning/spatial-contiguity-principle/0824E3ABE3556D77AADBE7714EFA7BCD) (the tests are in Moreno and Mayer, 1999, [Journal of Educational Psychology](https://doi.org/10.1037/0022-0663.91.2.358)): people learn better when the words sit next to the picture, because they stop searching the screen and can hold both in mind at once. A screen, a side-by-side, or a labeled figure needs a page. A call tree already puts the words on the nesting, so a page adds a click and no search gain.
- Mayer, coherence, in Moreno and Mayer, 2000, [Journal of Educational Psychology](https://doi.org/10.1037/0022-0663.92.1.117), and summarised in the [Cambridge Handbook, chapter 12](https://doi.org/10.1017/CBO9780511816819.013): extra words, pictures, and pages hurt. One file, one point, no navigation, no decoration. A multi-page site adds the search the page was meant to remove.
- The surface table above. A fence the host shows as source is not a view. The page is.

## How the page is generated

The page has to be one file, offline, with the words on the picture, and easy for the agent to get right. [MermaidSeqBench](https://arxiv.org/abs/2511.14967) is why "write the HTML yourself" loses: models already fail at a smaller syntax. These candidates drop out on that filter:

| Candidate | Why it drops out | Source |
| --- | --- | --- |
| Tailwind or Pico from a CDN | The page fetches the CSS when it opens. The file is not offline. | [Tailwind Play CDN](https://tailwindcss.com/docs/installation/play-cdn), [Pico CSS](https://picocss.com/) |
| reveal.js | A deck is a sequence of pages. Coherence, above, says extra pages add search. | [reveal.js](https://revealjs.com/) |
| D3 | The agent writes the layout. That is the hard part the spec was meant to remove. | [d3js.org](https://d3js.org/) |
| Hand-written HTML | Same failure mode as hand-written Mermaid, with more ways to break the page. | MermaidSeqBench, above |

`scripts/page.mjs` is the one that stays. The agent writes a JSON spec of the block kinds the skill already names (`screen`, `columns`, `tree`, `components`, `topology`, `diff`, `pre`, `mermaid`).

[Humanlayer's `show-me`](https://github.com/humanlayer/skills/blob/main/plugins/show-me/skills/show-me/SKILL.md) draws a component tree with the file on the component, and a file tree with a responsibility on each directory. An open suggestion on that skill, [pull request 4](https://github.com/humanlayer/skills/pull/4), adds a topology for parts that sit side by side over a shared layer, and uses color for the layer rather than decoration. `components`, a `tree` line written `name # note`, and `topology` are those three drawings. The markup is ours. The script inlines the CSS and, for a `mermaid` block, Mermaid's own browser build, the same `dist/mermaid.min.js` that `diagram.mjs` already checks and inlines. `examples/capabilities.json` is that spec for every kind. The script refuses the file when the spec is incomplete or Mermaid cannot parse a block.

`sketch-it` picks the view the reader can take in with the least search, then the diagram type, then the surface that actually draws it. A page is a spec, not a document the agent lays out.
