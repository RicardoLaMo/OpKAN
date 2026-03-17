import queue

# Thread-safe queues for dual-process communication

# System 1 (reflex / fast path): work context submitted to the worker
reflex_queue = queue.Queue(maxsize=1)   # Only keep the most-recent context

# System 2 (strategic / slow path): work context submitted to the worker
strategic_queue = queue.Queue(maxsize=1)

# Outbound decision queues populated by each worker
# Strategic is bounded to prevent memory growth if training halts or lags
reflex_decision_queue = queue.Queue(maxsize=32)
strategic_decision_queue = queue.Queue(maxsize=32)

# ---------------------------------------------------------------------------
# Legacy aliases kept for backwards-compatibility during migration
# ---------------------------------------------------------------------------
context_queue = reflex_queue
decision_queue = reflex_decision_queue
