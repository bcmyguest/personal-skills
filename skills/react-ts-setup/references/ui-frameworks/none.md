# None (plain CSS / CSS Modules)

Chosen UI path: **no component library**. (For the MUI default, see SKILL.md step 6.)
Ask the user for a basic colour set first — primary + secondary/accent at minimum,
ideally error and background.

Even with no framework, centralise the colours as CSS custom properties so the
palette lives in one place:

`src/styles/tokens.css` (imported once in `main.tsx`):

```css
:root {
  --color-primary: #1f6feb;
  --color-secondary: #a371f7;
  --color-error: #d1242f;
  --color-bg: #ffffff;
  --color-surface: #f6f8fa;
}
```

Use CSS Modules (`*.module.css`) for component-scoped styles and reference the
variables via `var(--color-primary)`. No provider, no extra dependencies.
