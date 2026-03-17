import threading
import queue
from typing import Dict, Any, Optional
from src.agent.core import LiuClawAgent
from src.models.mutator import TopologicalMutator
from src.engine.queues import (
    reflex_queue, strategic_queue,
    reflex_decision_queue, strategic_decision_queue,
)


class EngineCoordinator:
    """
    Dual-process coordinator for the async interaction between the PyTorch KAN
    training loop and the LiuClaw agent.

    System 1 (reflex / fast): handles edge pruning and LR adjustments.
    System 2 (strategic / slow): handles deliberate topological mutations and
                                  regime analysis.
    """

    def __init__(self, agent: LiuClawAgent):
        self.agent = agent
        self.running = False
        self._system_1_thread: Optional[threading.Thread] = None
        self._system_2_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_threads(self):
        """Starts the System 1 (reflex) and System 2 (strategic) worker threads."""
        self.running = True
        self._system_1_thread = threading.Thread(
            target=self._system_1_worker, daemon=True, name="system_1_reflex"
        )
        self._system_2_thread = threading.Thread(
            target=self._system_2_worker, daemon=True, name="system_2_strategic"
        )
        self._system_1_thread.start()
        self._system_2_thread.start()

    def stop_threads(self):
        """Stops both worker threads gracefully."""
        self.running = False
        if self._system_1_thread:
            self._system_1_thread.join(timeout=2.0)
        if self._system_2_thread:
            self._system_2_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------

    def _system_1_worker(self):
        """Fast-path worker: calls agent.think_fast() and posts a ReflexDecision."""
        while self.running:
            try:
                context = reflex_queue.get(timeout=0.1)
                step, edge_stats, loss_delta = context
                decision = self.agent.think_fast(step, edge_stats, loss_delta)
                reflex_decision_queue.put(decision)
                reflex_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[system_1] Error: {e}")

    def _system_2_worker(self):
        """Slow-path worker: calls agent.think_slow() and posts a StrategicDecision."""
        while self.running:
            try:
                context = strategic_queue.get(timeout=0.1)
                history, regime_data, model_state = context
                decision = self.agent.think_slow(history, regime_data, model_state)
                strategic_decision_queue.put(decision)
                strategic_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[system_2] Error: {e}")

    # ------------------------------------------------------------------
    # Submission helpers
    # ------------------------------------------------------------------

    def request_reflex(self, step: int, edge_stats: Dict[str, Any], loss_delta: float):
        """Submits a reflex (System 1) context for processing."""
        if reflex_queue.full():
            try:
                reflex_queue.get_nowait()
            except queue.Empty:
                pass
        reflex_queue.put((step, edge_stats, loss_delta))

    def request_strategic(
        self,
        history: Dict[str, Any],
        regime_data: Dict[str, Any],
        model_state: Dict[str, Any],
    ):
        """Submits a strategic (System 2) context for processing."""
        if strategic_queue.full():
            try:
                strategic_queue.get_nowait()
            except queue.Empty:
                pass
        strategic_queue.put((history, regime_data, model_state))

    # ------------------------------------------------------------------
    # Apply mutations
    # ------------------------------------------------------------------

    def apply_pending_mutations(self, model: Any, optimizer=None) -> str:
        """
        Drains both decision queues and applies any pending mutations to *model*.
        Returns "HALT" if a StrategicDecision requests it, otherwise "CONTINUE".
        """
        # --- System 1 (reflex) decisions ---
        while not reflex_decision_queue.empty():
            try:
                decision = reflex_decision_queue.get_nowait()
                for edge_id in decision.prunes:
                    TopologicalMutator.mutate_edge(model, edge_id, "PRUNE")
                if optimizer is not None and decision.lr_adjustment != 1.0:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= decision.lr_adjustment
                reflex_decision_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"[apply_reflex] Failed: {e}")

        # --- System 2 (strategic) decisions ---
        while not strategic_decision_queue.empty():
            try:
                decision = strategic_decision_queue.get_nowait()
                if decision.training_command == "HALT":
                    print("LiuClaw HALT command received! Terminating training loop.")
                    return "HALT"
                for mutation in decision.mutations:
                    TopologicalMutator.mutate_edge(
                        model,
                        mutation.edge_id,
                        mutation.action,
                        mutation.formula,
                        mutation.initial_params,
                    )
                strategic_decision_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"[apply_strategic] Failed: {e}")

        return "CONTINUE"
