# Mantine

Chosen UI path: **Mantine**. (For the MUI default, see SKILL.md step 6.) Ask the user
for a basic colour set first — primary + secondary/accent at minimum, ideally error
and background — and put them in the theme object so a rebrand is a one-file change.

Mantine is a component library with a provider + theme object, the same shape as the
MUI example in SKILL.md step 6.

```bash
pnpm add @mantine/core @mantine/hooks
```

Wrap the app in `<MantineProvider theme={theme}>` and import `@mantine/core/styles.css`.
Define the colours in the theme object exactly as the MUI example defines `palette` —
one module, referenced everywhere, never hard-coded hex in a component.
