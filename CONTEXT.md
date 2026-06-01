# 当前状态（2026-06-01）

- 当前正在做什么：分析 Claude Code 的记忆与上下文设计，评估 nanobot 适合的改进方向。
- 上次停在哪个位置：已对照 `ContextBuilder`、`MemoryStore`、`EpisodeStore`、`SkillsLoader`，初步结论是 nanobot 已有基础三层记忆，但分类、召回和压缩策略还偏粗。
- 近期关键决定和原因：优先考虑低成本改进，不照搬 Claude Code 的复杂生产级机制。
