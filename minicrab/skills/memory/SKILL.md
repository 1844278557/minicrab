---
name: memory
description: Two-file memory system for user profile memory and durable working memory.
always: true
---

# Memory

## Structure

- `USER.md` — Stable user identity, preferences, personal facts, and long-term behavior requirements. Loaded into context.
- `memory/MEMORY.md` — Durable cross-session experience, business rules, long-term project decisions, and important external information locations. Loaded into context while small.
- `memory/episodes/` — Legacy archive of older conversation summaries. Not a default memory source.

## Routing Rules

- Use `USER.md` for facts about the user: identity, preferences, relationships, dates, language, timezone, and how the assistant should behave for this user.
- Use `memory/MEMORY.md` for reusable work knowledge: business rules, durable project decisions, lessons learned, and where important external information can be found.
- In `memory/MEMORY.md`, keep one memory per bullet and use `[rule][YYYY-MM-DD]`, `[decision][YYYY-MM-DD]`, `[lesson][YYYY-MM-DD]`, or `[info][YYYY-MM-DD]` under the matching section.
- Use `CONTEXT.md` for the current task state: goal, active step, completed steps, next step, blockers, decisions, validation, and cautions.
- Use RAG for uploaded documents and reference files that the user wants to query later.
- Do not put raw tool output, document contents, temporary chat, or current task progress into `USER.md` or `MEMORY.md`.

## Explicit Memory Requests

When the user explicitly asks to remember something, choose the destination by content:

- Personal/user-level information → `USER.md`
- Business rules, reusable project knowledge, or long-term decisions → `memory/MEMORY.md`
- Unclear destination → ask one short clarification before writing
- Use `manage_memory` for new remember, correct, and forget requests.
- Use `append` for new memory, `replace` for explicit corrections or new decisions that supersede old ones, and `delete` when the user asks to forget something.
- For `replace` and `delete`, first read the target memory file and pass the exact old bullet as `match`.

## Episode Archive

Use episode tools only when the user explicitly asks about old conversations, previous discussions, or historical context that is not already present in `USER.md`, `MEMORY.md`, `CONTEXT.md`, or recent chat history.

## Auto-consolidation

Old conversations may be compacted when the session grows large. Treat compaction as session continuity, not as a reason to store everything in long-term memory.
