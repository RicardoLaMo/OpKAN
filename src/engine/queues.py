import queue

# Thread-safe queues for async communication

# System 1 (reflex / fast path): work context submitted to the worker
reflex_queue = queue.Queue(maxsize=1)   # Only keep the most-recent context

# System 2 (strategic / slow path): work context submitted to the worker
strategic_queue = queue.Queue(maxsize=1)

# Outbound decision queues populated by each worker
reflex_decision_queue = queue.Queue()
strategic_decision_queue = queue.Queue()

# ---------------------------------------------------------------------------
# Legacy aliases kept for backwards-compatibility during migration
# ---------------------------------------------------------------------------
context_queue = reflex_queue
decision_queue = reflex_decision_queue
