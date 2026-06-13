# Tailwind CSS v4

Chosen UI path: **Tailwind**. (For the MUI default, see SKILL.md step 6.) Ask the
user for a basic colour set first — primary + secondary/accent at minimum, ideally
error and background — and define them as theme tokens in one place so a rebrand is
a one-file change.

Tailwind v4 is configured through the Vite plugin and a CSS-first `@theme` block —
there is no `tailwind.config.js` for the common case.

```bash
pnpm add tailwindcss @tailwindcss/vite
```

Add the plugin in `vite.config.ts` (alongside `react()` and the React-compiler babel
plugin from SKILL.md step 4):

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
