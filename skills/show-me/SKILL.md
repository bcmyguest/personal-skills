---
name: show-me
description: Point the user to the code or source behind the previous response, with a snippet plus a link to the exact lines, a link to the exact section of the source material, or both. Use when the user runs /show-me or asks where something you said comes from.
argument-hint: "[item numbers, comma-separated] [what to locate]"
---

# Show me

Show the user the evidence behind your last response, so they can check it themselves.

## Arguments

`$ARGUMENTS` holds up to two parts, both optional:

- **Item numbers,** comma-separated (`1,3`). They refer to the numbered items in your last response. When it had no numbered items, count its top-level bullets or paragraphs from 1, and state the mapping in one line.
- **Target,** any text after the numbers ("the retry logic", "the rate-limit docs"). It names what to locate.

No arguments: cover every concrete claim in the last response about code, behaviour, docs, or data.

## For each claim

| The claim rests on | Give |
| --- | --- |
| Code in this workspace | The snippet as a fenced block, headed by `path:start-end`. |
| Code in a pushed commit | The snippet plus a permalink: `https://github.com/<owner>/<repo>/blob/<sha>/<path>#L<start>-L<end>`. |
| Docs, a spec, a paper, an issue | A link to the exact section, plus the one or two sentences that support the claim. |
| Code and a source | Both. |

## Rules

1. Locate before you link. Read the file or fetch the page now; a snippet or quote comes from what you read, never from memory.
2. Show the smallest span that proves the claim: usually 3 to 15 lines.
3. Build permalinks from `git rev-parse HEAD` and `git remote get-url origin`. When the lines have uncommitted changes, or the commit is not pushed, give `path:line` only and say why.
4. Link the primary source (official docs, spec, source code, the paper itself), not a summary of it.
5. When you cannot find support for a claim, say "not found" and retract or qualify the claim. Never invent a path, line, or URL.
6. Do not re-explain. One line per item saying what the evidence shows, then the evidence.
