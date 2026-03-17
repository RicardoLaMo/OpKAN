import queue

# Thread-safe queues for async communication
# Context: (kan_stats, current_regime, vol_info)
context_queue = queue.Queue(maxsize=1) # Only need the most recent context

# Decisions: LiuClawDecision
decision_queue = queue.Queue() # Collect all proposed mutations
