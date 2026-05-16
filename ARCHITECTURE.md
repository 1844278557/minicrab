# Architecture

## Overall Flow

Nanobot routes every chat channel through the same message bus:

```text
Channel -> InboundMessage -> MessageBus -> AgentLoop -> OutboundMessage -> Channel
```

This keeps platform integrations separate from the agent core.

## Key Modules

- `nanobot/agent/loop.py`: Runs the agent, builds context, calls the model, executes tools, and writes replies back to the bus.
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

