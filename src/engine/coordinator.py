import threading
import queue
from typing import Any, Dict
from src.agent.core import LiuClawAgent
from src.models.mutator import TopologicalMutator
from src.engine.queues import (
    reflex_queue, strategic_queue, 
    reflex_decision_queue, strategic_decision_queue
)

class EngineCoordinator:
    """
    Coordinates the Dual-Process (Fast and Slow) thinking loops.
    """
    def __init__(self, agent: LiuClawAgent):
        self.agent = agent
        self.running = False
        self.system_1_thread = None
        self.system_2_thread = None

    def start_threads(self):
        self.running = True
        self.system_1_thread = threading.Thread(target=self._system_1_worker, daemon=True)
        self.system_2_thread = threading.Thread(target=self._system_2_worker, daemon=True)
        self.system_1_thread.start()
        self.system_2_thread.start()

    def stop_threads(self):
        self.running = False
        if self.system_1_thread: self.system_1_thread.join(timeout=1.0)
        if self.system_2_thread: self.system_2_thread.join(timeout=1.0)

    def _system_1_worker(self):
        """Low-latency reflexive thinking."""
        while self.running:
            try:
                context = reflex_queue.get(timeout=0.1)
                decision = self.agent.think_fast(context['step'], context['edge_stats'], context['loss_delta'])
                reflex_decision_queue.put(decision)
                reflex_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"System 1 Error: {e}")

    def _system_2_worker(self):
        """High-latency strategic thinking."""
        while self.running:
            try:
                context = strategic_queue.get(timeout=0.1)
                decision = self.agent.think_slow(context['history'], context['regime_data'], context['model_state'])
                strategic_decision_queue.put(decision)
                strategic_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"System 2 Error: {e}")

    @staticmethod
    def apply_pending_mutations(model: Any, optimizer: Any = None):
        """Processes decisions from both fast and slow thinking loops."""
        # 1. Process Reflexive (Fast) Decisions
        while not reflex_decision_queue.empty():
            decision = reflex_decision_queue.get_nowait()
            print(f"⚡ Reflexive Action: {decision.reasoning}")
            for edge_id in decision.prunes:
                TopologicalMutator.mutate_edge(model, edge_id, "PRUNE")
            if optimizer and decision.lr_adjustment != 1.0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decision.lr_adjustment
            reflex_decision_queue.task_done()

        # 2. Process Strategic (Slow) Decisions
        while not strategic_decision_queue.empty():
            decision = strategic_decision_queue.get_nowait()
            print(f"🧠 Strategic Review: {decision.reasoning}")
            for mutation in decision.mutations:
                TopologicalMutator.mutate_edge(model, mutation.edge_id, mutation.action, mutation.formula, mutation.initial_params)
            if decision.regime_analysis.hmm_transition_detected:
                print(f"🚨 Strategic Regime Shift: {decision.regime_analysis.predicted_regime}")
            strategic_decision_queue.task_done()

    @staticmethod
    def request_reflex(step: int, edge_stats: dict, loss_delta: float):
        if not reflex_queue.full():
            reflex_queue.put({'step': step, 'edge_stats': edge_stats, 'loss_delta': loss_delta})

    @staticmethod
    def request_strategic(history: dict, regime_data: dict, model_state: dict):
        if strategic_queue.empty(): # Only one strategic review at a time
            strategic_queue.put({'history': history, 'regime_data': regime_data, 'model_state': model_state})
