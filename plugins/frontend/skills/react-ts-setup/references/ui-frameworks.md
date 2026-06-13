# UI frameworks and theming

SKILL.md wires **Material UI (MUI)** as the worked example. This file covers the
other choices. Whatever the framework, **ask the user for a basic colour set
before writing the theme** — at minimum a primary and secondary/accent colour;
ideally also error/success and background. Put those colours in one theme/tokens
module and reference them everywhere, so a rebrand is a one-file change rather
than a find-and-replace.

A barebones starting palette (replace with the user's brand colours):

| Token      | Hex       | Use                          |
| ---------- | --------- | ---------------------------- |
| primary    | `#1f6feb` | buttons, links, active state |
| secondary  | `#a371f7` | accents, highlights          |
| error      | `#d1242f` | validation, destructive      |
| background | `#ffffff` | page background              |
| surface    | `#f6f8fa` | cards, raised surfaces       |

---

## Material UI (MUI) — installed in SKILL.md

```bash
pnpm add @mui/material @emotion/react @emotion/styled
# optional: pnpm add @mui/icons-material @fontsource/roboto
```

`src/theme.ts`, then wrap the app in `ThemeProvider` + `CssBaseline` (see SKILL.md
step 6). Components read colours via `theme.palette.*` — never hard-code hex in a
component.

---

## Tailwind CSS v4

Tailwind v4 is configured through the Vite plugin and a CSS-first `@theme` block —
there is no `tailwind.config.js` for the common case.

```bash
pnpm add tailwindcss @tailwindcss/vite
```

Add the plugin in `vite.config.ts` (alongside `react()`):

```ts
import tailwindcss from '@tailwindcss/vite'
// plugins: [react(), tailwindcss(), babel({ presets: [reactCompilerPreset()] })]
```

`src/index.css` — import Tailwind and define the palette as theme tokens, which
become utilities like `bg-primary` / `text-primary`:

```css
@import 'tailwindcss';

@theme {
  --color-primary: #1f6feb;
  --color-secondary: #a371f7;
  --color-error: #d1242f;
}
```

No `ThemeProvider` is needed. Recommend the **Tailwind CSS IntelliSense**
(`bradlc.vscode-tailwindcss`) and **Prettier plugin for Tailwind**
(`prettier-plugin-tailwindcss`, sorts class names) — add the latter to
`devDependencies` and `.prettierrc.json`'s `plugins`.

---

## Chakra UI / Mantine

Both are component libraries with a provider + theme object, same shape as MUI:

- **Chakra**: `pnpm add @chakra-ui/react @emotion/react`; wrap in
  `<ChakraProvider value={system}>` where `system` comes from `createSystem`.
- **Mantine**: `pnpm add @mantine/core @mantine/hooks`; wrap in
  `<MantineProvider theme={theme}>`; import `@mantine/core/styles.css`.

Define colours in the theme object exactly as the MUI example does.

---

## None (plain CSS / CSS Modules)

When the user wants no component library, still centralise colours as CSS custom
properties so the palette lives in one place:

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
