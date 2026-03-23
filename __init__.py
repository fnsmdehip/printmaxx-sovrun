"""sovrun core — meta-cognition framework for AI agents.

stop dog walking your AI through every decision.
6.9 corrections per task → 1.

Modules:
    voice_extractor    - analyze prompt history, distill communication style
    cognitive_engine   - build cognition model from correction chains + task history
    pattern_miner      - extract meta-rules from user prompt patterns
    user_sim_refiner   - simulate user critique for autonomous improvement
    loop_closer        - close open loops: decide, act, log, learn
    self_audit         - competitive cognition audit (meta-improvement)
    decision_engine    - closed-loop autonomous decision agent
    resilience         - retry, file locking, circuit breaker, sanitization, parallel guardrails
    conversation_logger - extract and search conversation transcripts
    memory_import      - import conversation history from any LLM provider
    session_briefing   - generate session start briefings from system state
    handoff            - typed agent-to-agent handoff protocol with guardrails
    procedural_memory  - skill document system: capture, recall, consolidate learned tasks
    orchestration      - DAG-based step execution with parallel fanout, checkpointing, resume
    durable            - crash recovery with deterministic replay
    tracing            - agent observability: event collection, cost tracking, HTML timelines
    mcp_bridge         - expose agents as MCP tool servers, consume external MCP tools
    deps               - optional dependency upgrades with stdlib fallback (tenacity, httpx, rich)
    workflow_bridge     - autonomous n8n workflow creation and management via API
    conversation_index - FTS5 full-text search index for conversation history
    media              - smart media generation router (image, video, TTS, voice agent, music)
"""

__version__ = "0.1.0"
