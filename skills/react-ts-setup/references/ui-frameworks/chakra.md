# Chakra UI

Chosen UI path: **Chakra**. (For the MUI default, see SKILL.md step 6.) Ask the user
for a basic colour set first — primary + secondary/accent at minimum, ideally error
and background — and put them in the theme object so a rebrand is a one-file change.

Chakra is a component library with a provider + theme object, the same shape as the
MUI example in SKILL.md step 6.

```bash
pnpm add @chakra-ui/react @emotion/react
```

Wrap the app in `<ChakraProvider value={system}>`, where `system` comes from
`createSystem`. Define the colours in the system/theme config exactly as the MUI
example defines `palette` — one module, referenced everywhere, never hard-coded hex
in a component.
