import queue

# Thread-safe queues for async communication
context_queue = queue.Queue(maxsize=1)   # Only need the most recent context

# Bounded to prevent unbounded memory growth if mutations queue faster than consumed
decision_queue = queue.Queue(maxsize=32)
