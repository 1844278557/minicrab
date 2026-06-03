# Current Task State

Use this file as the structured task state block. Keep every heading, write short bullets, and update it after meaningful project work.

## Current Goal

Redesign memory boundaries on the `memory-redesign-v1` branch.

## Active Step

Step 2 completed: route explicit memory updates to USER.md or MEMORY.md.

## Completed Steps

- Added memory routing rules.
- Added current task context loading and lightweight context budget trimming.
- Clarified `CONTEXT.md` as a structured task state block.
- Added `ToolResultSummarizer` for long string tool outputs.
- Connected summarization after tool execution and before tool results enter the next LLM context.
- Created `memory-redesign-v1` from `memory-routing-rules`.
- Clarified memory boundaries in prompts, skills, and templates.
- Added `target` routing to `remember_user_info` for USER.md and memory/MEMORY.md.
- Updated approval policy wording for both long-term user memory and long-term working memory.

## Next Step

- Step 3: add post-response automatic extraction into MEMORY.md only.

## Blockers

- None.

## Key Decisions

- Do not implement automatic history recall in this branch.
- Keep the task state structure simple and prompt-driven instead of adding a new task-state database.
- Tool result summarization is an internal no-tool LLM call, not an Agent or subagent.
- Short tool results remain unchanged; long result summary failure falls back to deterministic head/tail truncation.
- USER.md is for user-level memory; MEMORY.md is for durable work memory; RAG is for uploaded documents; episodes are legacy archives.
- Explicit user memory writes keep the existing `remember_user_info` tool name and use `target=user|memory` for routing.

## Validation

- Context prompt tests: 10 passed.
- Context + user memory tests: 16 passed.
- Syntax check: passed.
- Runner tests: 10 passed.
- Runner + context + save-turn tests: 24 passed.
- Tool summarizer syntax check: passed.
- Step 1 context prompt tests: 10 passed.
- Step 2 user memory + approval tests: 20 passed.

## Cautions

- Do not merge this branch into `main` until the user reviews it.
- Leave unrelated `docs/NANOBOT_INTERVIEW_QA.md` changes uncommitted.
