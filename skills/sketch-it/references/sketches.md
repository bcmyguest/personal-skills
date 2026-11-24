# Sketch shapes

Read the one section `SKILL.md` step 1 selected. Keep only the calls, files, props, states, and boundaries that answer the current question.

## Pseudocode

```text
on(request)
  if token is expired
    refresh token
  call upstream
  return response
```

## Call tree

Indentation is the call. No arrows.

```text
handleRequest
  authenticate
    loadSession
  dispatch
    queryStore
```

## Component tree

Name the file that owns a component the first time it appears. Include a hook or prop only when it is part of the point.

```tsx
<SettingsPage> (src/routes/settings.tsx)
  useSettings()
  <SettingsForm> (src/components/settings-form.tsx)
    <SaveButton>
```

## File tree

One or two levels. One responsibility comment per directory.

```text
src/
├── routes/        # page entry, loads data
├── components/    # renders one screen region
└── api/           # talks to the server
```

## Shape diff

Diff the same shape the reader already has. Match the diff to the topic: component tree, file tree, call tree, or pseudocode.

```diff
 handleRequest
   authenticate
     loadSession
   dispatch
+    authorize
     queryStore
```

## Whole block

The block as it is in the file now, with `path:line` above the fence. Use this when most of the block is new, when a diff would hide ownership or order, or when the reader needs a copyable target.

## HTML page

Write a JSON spec. Do not write HTML. `scripts/page.mjs` turns the spec into one offline file: the CSS is inlined, a `mermaid` block is checked and inlined, and there is no navigation.

```json
{
  "title": "Where the token is checked",
  "point": "The check happens before dispatch.",
  "blocks": [
    {
      "kind": "screen",
      "regions": [
        { "name": "Request", "note": "checked in authenticate, before dispatch" }
      ]
    }
  ]
}
```

`title` is required. `point` is one sentence. Each block may have its own `title`. Kinds:

| kind | Fields | Use |
| --- | --- | --- |
| `screen` | `regions`: `{ name, note? }` | Where regions sit. The note is inside the region. |
| `columns` | `items`: at least two `{ title, text }` | Two appearances, side by side. They stack on a narrow screen. |
| `tree` | `text` | Indented names, two spaces per level. |
| `diff` | `text` | A line that starts with `+` or `-` is the change. |
| `pre` | `text` | Pseudocode or a file tree, kept monospaced. |
| `mermaid` | `source` | A diagram. The script refuses the page if Mermaid cannot parse it. |

A full page that uses every kind is `examples/capabilities.json`.
