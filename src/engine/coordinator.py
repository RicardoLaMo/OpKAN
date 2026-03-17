import threading
import time
import queue
from typing import Dict, Any, Callable, Optional
from src.agent.core import LiuClawAgent
from src.models.mutator import TopologicalMutator
from src.engine.queues import context_queue, decision_queue

# Separate queues for dual-process (reflex / strategic) architecture
_reflex_queue: queue.Queue = queue.Queue(maxsize=10)
_strategic_queue: queue.Queue = queue.Queue(maxsize=10)
_pending_decisions: queue.Queue = queue.Queue()


class EngineCoordinator:
    """
    Coordinates the async interaction between the PyTorch KAN training loop
    and the LiuClaw agent.

    Supports two interaction modes:
      - Legacy single-thread: start_agent_thread / stop_agent_thread / request_mutation
      - Dual-process: start_threads / stop_threads / request_reflex / request_strategic
    """
    def __init__(self, agent: LiuClawAgent):
        self.agent = agent
        self.running = False
        self.agent_thread = None
        self._reflex_thread: Optional[threading.Thread] = None
        self._strategic_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Legacy API (preserved for backward compatibility)
    # ------------------------------------------------------------------

    def start_agent_thread(self):
        """Starts the background thread for the agent's reasoning loop."""
        self.running = True
        self.agent_thread = threading.Thread(target=self._agent_worker, daemon=True)
        self.agent_thread.start()

    def stop_agent_thread(self):
        """Stops the agent worker."""
        self.running = False
        if self.agent_thread:
            self.agent_thread.join(timeout=1.0)

    def _agent_worker(self):
        """Worker function for the agent reasoning loop."""
        while self.running:
            try:
                context = context_queue.get(timeout=0.1)
                kan_state, pipeline_health = context
                decision = self.agent.decide_mutations(kan_state, pipeline_health)
                decision_queue.put(decision)
                context_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in agent worker thread: {e}")

    @staticmethod
    def request_mutation(kan_state: Dict[str, Any], pipeline_health: Dict[str, Any]):
        """Submits context to the agent for decision making."""
        if context_queue.full():
            try:
                context_queue.get_nowait()
            except queue.Empty:
                pass
        context_queue.put((kan_state, pipeline_health))

    # ------------------------------------------------------------------
    # Dual-process API
    # ------------------------------------------------------------------

    def start_threads(self):
        """Starts both reflex (fast) and strategic (slow) background threads."""
        self.running = True
        self._reflex_thread = threading.Thread(target=self._reflex_worker, daemon=True)
        self._strategic_thread = threading.Thread(target=self._strategic_worker, daemon=True)
        self._reflex_thread.start()
        self._strategic_thread.start()

    def stop_threads(self):
        """Stops both background threads."""
        self.running = False
        if self._reflex_thread:
            self._reflex_thread.join(timeout=1.0)
        if self._strategic_thread:
            self._strategic_thread.join(timeout=1.0)

    def request_reflex(self, step: int, edge_stats: Dict[str, Any], loss_delta: float):
        """Submits a fast-path reflex request to the agent."""
        if _reflex_queue.full():
            try:
                _reflex_queue.get_nowait()
            except queue.Empty:
                pass
        _reflex_queue.put({"step": step, "edge_stats": edge_stats, "loss_delta": loss_delta})

    def request_strategic(self, history: Dict[str, Any], regime_data: Dict[str, Any], model_state: Dict[str, Any]):
        """Submits a slow-path strategic request to the agent."""
        if _strategic_queue.full():
            try:
                _strategic_queue.get_nowait()
            except queue.Empty:
                pass
        _strategic_queue.put({"history": history, "regime_data": regime_data, "model_state": model_state})

    def _reflex_worker(self):
        """Background thread that calls agent.think_fast for low-latency decisions."""
        while self.running:
            try:
                ctx = _reflex_queue.get(timeout=0.1)
                decision = self.agent.think_fast(
                    step=ctx["step"],
                    edge_stats=ctx["edge_stats"],
                    loss_delta=ctx["loss_delta"],
                )
                _pending_decisions.put(("reflex", decision))
                _reflex_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in reflex worker thread: {e}")

    def _strategic_worker(self):
        """Background thread that calls agent.think_slow for full mutation decisions."""
        while self.running:
            try:
                ctx = _strategic_queue.get(timeout=0.1)
                decision = self.agent.think_slow(
                    history=ctx["history"],
                    regime_data=ctx["regime_data"],
                    model_state=ctx["model_state"],
                )
                _pending_decisions.put(("strategic", decision))
                _strategic_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in strategic worker thread: {e}")

    # ------------------------------------------------------------------
    # Mutation application (supports both APIs)
    # ------------------------------------------------------------------

    def apply_pending_mutations(self, model: Any, optimizer: Any = None):
        """
        Checks pending decision queues and applies any ready mutations.
        Handles both legacy (decision_queue) and dual-process (_pending_decisions) decisions.
        """
        # Dual-process decisions
        while not _pending_decisions.empty():
            try:
                kind, decision = _pending_decisions.get_nowait()
                if kind == "strategic":
                    training_cmd = getattr(decision, "training_command", "CONTINUE")
                    if training_cmd == "HALT":
                        print("LiuClaw HALT command received! Terminating training loop.")
                        return "HALT"
                    print(f"Applying strategic mutations. Reasoning: {decision.reasoning}")
                    for mutation in getattr(decision, "mutations", []):
                        status = TopologicalMutator.mutate_edge(
                            model,
                            mutation.edge_id,
                            mutation.action,
                            mutation.formula,
                            mutation.initial_params,
                        )
                        print(f"Mutation status: {status}")
                    regime = getattr(decision, "regime_analysis", None)
                    if regime and regime.hmm_transition_detected:
                        print(f"Regime Shift Detected: {regime.predicted_regime}!")
                        print(f"Thesis: {regime.thesis_statement}")
                elif kind == "reflex":
                    lr_adj = getattr(decision, "lr_adjustment", 1.0)
                    if lr_adj != 1.0 and optimizer is not None:
                        for pg in optimizer.param_groups:
                            pg["lr"] *= lr_adj
                        print(f"Reflex LR adjustment: x{lr_adj:.4f}")
                _pending_decisions.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Failed to apply decision: {e}")

        # Legacy decisions
        while not decision_queue.empty():
            try:
                decision = decision_queue.get_nowait()
                if decision.training_command == "HALT":
                    print("LiuClaw HALT command received! Terminating training loop.")
                    return "HALT"
                print(f"Applying mutations from agent. Reasoning: {decision.reasoning}")
                for mutation in decision.mutations:
                    status = TopologicalMutator.mutate_edge(
                        model,
                        mutation.edge_id,
                        mutation.action,
                        mutation.formula,
                        mutation.initial_params,
                    )
                    print(f"Mutation status: {status}")
                if decision.regime_analysis.hmm_transition_detected:
                    print(f"Regime Shift Detected: {decision.regime_analysis.predicted_regime}!")
                    print(f"Thesis: {decision.regime_analysis.thesis_statement}")
                decision_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Failed to apply mutation: {e}")
        return "CONTINUE"
