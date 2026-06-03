# Current Task State

Use this file as the structured task state block. Keep every heading, write short bullets, and update it after meaningful project work.

## Current Goal

Redesign memory boundaries on the `memory-redesign-v1` branch.

## Active Step

Step 5 completed: keep episode tools as legacy archive only.

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
- Added `MemoryExtractor` to scan newly saved turn messages after replies.
- Connected post-turn background maintenance to run memory extraction before token consolidation.
- Added `memory_extracted_until` session metadata to avoid rescanning old messages.
- Changed token compaction to produce `session.metadata.running_summary`.
- Stopped token compaction from writing MEMORY.md, HISTORY.md, or new episode files.
- Added current session summary injection to LLM context.
- Updated episode tool descriptions so they are only used for explicit old-history archive requests.

## Next Step

- Step 6: update README.md and ARCHITECTURE.md, then run final regression tests.

## Blockers

- None.

## Key Decisions

- Do not implement automatic history recall in this branch.
- Keep the task state structure simple and prompt-driven instead of adding a new task-state database.
- Tool result summarization is an internal no-tool LLM call, not an Agent or subagent.
- Short tool results remain unchanged; long result summary failure falls back to deterministic head/tail truncation.
- USER.md is for user-level memory; MEMORY.md is for durable work memory; RAG is for uploaded documents; episodes are legacy archives.
- Explicit user memory writes keep the existing `remember_user_info` tool name and use `target=user|memory` for routing.
- Automatic post-response extraction writes only MEMORY.md and never USER.md.
- Invalid extraction JSON is skipped and the scan cursor advances to avoid repeated retries on the same messages.
- Token compaction is for current-session continuity only; long-term memory extraction is handled separately after turns.
- Episode tools remain registered for legacy archive lookup, but are not default memory or session-summary sources.

## Validation

- Context prompt tests: 10 passed.
- Context + user memory tests: 16 passed.
- Syntax check: passed.
- Runner tests: 10 passed.
- Runner + context + save-turn tests: 24 passed.
- Tool summarizer syntax check: passed.
- Step 1 context prompt tests: 10 passed.
- Step 2 user memory + approval tests: 20 passed.
- Step 3 memory extractor + loop save tests: 9 passed.
- Step 3 combined related tests: 27 passed.
- Step 4 compaction target tests: 69 passed.
- Step 4 related regression tests: 35 passed.
- Step 4 AST syntax check: passed.
- Step 5 episode + context tests: 45 passed.

## Cautions

- Do not merge this branch into `main` until the user reviews it.
- Leave unrelated `docs/NANOBOT_INTERVIEW_QA.md` changes uncommitted.
