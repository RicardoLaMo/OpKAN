import queue

# Thread-safe queues for dual-process communication
reflex_queue = queue.Queue(maxsize=5)    # System 1: Fast tasks
strategic_queue = queue.Queue(maxsize=1) # System 2: Slow strategic tasks

# Decision queues
reflex_decision_queue = queue.Queue()
strategic_decision_queue = queue.Queue()
