---
name: match-ui-to-reference
description: Converge a UI on a design reference through numbered deltas and visual proof. Use when given a Figma link, mockup, or screenshot to match, or told the UI "doesn't match the design". Not for deciding what the product should show, and not for backend changes.
---

# Match UI to reference

Turn the gap between reference and app into a numbered list, close each item, and prove it with a screenshot.

## Gather first

- The reference (Figma node, image) and the running app at the same route and state.
- Target viewport or display, locales, and the project's i18n mechanism.

## Method

1. **List deltas, numbered:** layout, spacing, type, colour, copy, states, motion. One observable difference per item.
2. **Close items in order** and re-check each against the reference, not against memory of it.
3. **Verify by artifact:** a screenshot or recording of the running app at the target viewport. "Deployed" is not evidence; a stale build or cache is the usual cause of "nothing changed".
4. **Localise every new user-visible string,** including statuses, tooltips, and empty states, in every shipped locale.
5. **Use the platform:** the project's component library, i18n library, and animation primitives before hand-rolled code. No raw HTML injection for formatting.
6. **Judge motion by watching it:** order, stagger, and delays in a recording, not by reading the code.

## Deliverable

The delta list with each item marked done or deferred, before and after screenshots, and the list of strings added per locale.

## Boundaries

Edit frontend code and assets. Design changes that alter what is shown go back to the user.

## Next

- `audit-agent-diff` (transition): screenshots and the resolved delta list.
