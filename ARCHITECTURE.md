# Architecture

## Overall Flow

Nanobot routes every chat channel through the same message bus:

```text
Channel -> InboundMessage -> MessageBus -> AgentLoop -> OutboundMessage -> Channel
```

This keeps platform integrations separate from the agent core.

## Key Modules

- `nanobot/agent/loop.py`: Runs the agent, builds context, calls the model, executes tools, and writes replies back to the bus.
- `nanobot/agent/approval.py`: Decides whether risky tool calls need user approval and tracks pending confirmations in memory.
- `nanobot/agent/context.py`: Builds the system prompt from bootstrap files, long-term working memory, current session summary, skills, history, and runtime metadata.
- `nanobot/agent/memory.py`: Handles token-triggered session compaction into `session.metadata.running_summary` and provides long-term memory file helpers.
- `nanobot/agent/memory_extractor.py`: After each normal reply, extracts durable working memory into `memory/MEMORY.md` without writing user-profile memory.
- `nanobot/agent/tools/memory.py`: Updates `USER.md` or `memory/MEMORY.md` when the user explicitly asks the agent to remember something.
- `nanobot/bus/queue.py`: Provides the inbound and outbound queues used by channels and the agent.
- `nanobot/bus/events.py`: Defines the message objects passed through the bus.
- `nanobot/channels/base.py`: Defines the common channel interface.
- `nanobot/channels/manager.py`: Loads enabled channels and routes outbound messages to the right channel.
- `nanobot/channels/registry.py`: Discovers built-in channel modules and external channel plugins.
- `nanobot/channels/web.py`: Runs the local Web chat channel with FastAPI, receives browser messages over WebSocket, and sends agent replies back to the browser.
- `nanobot/templates/web/index.html`: Provides the Web chat page.
- `nanobot/templates/web/style.css`: Styles the Web chat page.
- `nanobot/templates/web/app.js`: Manages the browser WebSocket connection and message rendering.
- `nanobot/config/schema.py`: Defines shared configuration models; channel-specific configs are stored as extra fields under `channels`.

## Web Channel Design

The Web channel is a normal channel implementation, not a special agent mode.

```text
Browser
  -> WebSocket /web/ws
  -> WebChannel
  -> MessageBus
  -> AgentLoop
  -> MessageBus
  -> WebChannel
  -> Browser
```

First-version decisions:

- The page is served at `/web`.
- Browser messages are sent to `/web/ws`.
- The default chat id is `default`, so the session key is `web:default`.
- The channel defaults to `127.0.0.1` to avoid accidental public exposure.
- The frontend uses plain HTML, CSS, and JavaScript to avoid a build step.
- The browser requests the latest server-side session history over WebSocket and displays the most recent 100 visible messages.
- Risky actions are sent to Web as control events and rendered as modal approvals; other channels use the same text confirmation flow.

## Approval Design

- Approval is checked before tool execution inside the shared agent runner.
- First-version approval is in memory only; pending approvals are lost if the gateway restarts.
- `remember_user_info`, direct writes to root `USER.md`, risky shell commands, sensitive file writes, cron changes, RAG deletion/cleanup, media sends, and cross-chat sends require confirmation.
- Approval prompts are not saved as normal assistant messages, so they do not appear in Web history.
- After the user approves, the saved pending tool call is resumed and the model continues to produce a normal final reply.
- After the user rejects, the model continues without tool access, so it cannot use another tool to perform the rejected action.
- Web renders approval metadata as a modal dialog, while other channels receive a normal text prompt.

## Memory Design

- `USER.md` stores stable user identity, preferences, long-term personal facts, and user-level behavior requirements. It is loaded as a bootstrap file.
- `memory/MEMORY.md` stores durable cross-session working memory such as business rules, long-term project decisions, lessons learned, and important external information locations.
- The `remember_user_info` tool keeps its existing name but routes explicit user-requested writes with `target=user` or `target=memory`.
- After a normal reply is saved, `MemoryExtractor` scans only the newly saved turn and may update `memory/MEMORY.md`. It never writes `USER.md`.
- Token-triggered session compaction writes `session.metadata.running_summary` only. It does not update `USER.md`, `MEMORY.md`, `HISTORY.md`, or episode files.
- `ContextBuilder` loads `USER.md`, `memory/MEMORY.md`, `CONTEXT.md`, and the current session `running_summary` into the model context.
- Episode tools remain registered for legacy archive lookup, but they are not a default memory source.

## RAG Design

- RAG documents are stored in ChromaDB collections under `memory/rag/chroma`.
- The default embedding model keeps the legacy `rag_documents` collection.
- Non-default embedding models use model-specific collection and registry names, so switching from the default embedding to SiliconFlow/Qwen does not conflict with the old Chroma collection configuration.
