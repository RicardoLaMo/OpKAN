import copy
import threading
import time
import queue
from typing import Dict, Any, Callable, Optional
from src.agent.core import LiuClawAgent
from src.models.mutator import TopologicalMutator
from src.engine.queues import (
    reflex_queue, strategic_queue, 
    reflex_decision_queue, strategic_decision_queue,
    context_queue, decision_queue
)

class EngineCoordinator:
    """
    Coordinates the Dual-Process (Fast and Slow) thinking loops.
    Supports both legacy and dual-process interaction modes.
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
                
                # Bounded queue handling with drop-oldest strategy
                try:
                    decision_queue.put_nowait(decision)
                except queue.Full:
                    try:
                        decision_queue.get_nowait()
                        decision_queue.task_done()
                    except queue.Empty:
                        pass
                    decision_queue.put_nowait(decision)

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
        if self._reflex_thread: self._reflex_thread.join(timeout=1.0)
        if self._strategic_thread: self._strategic_thread.join(timeout=1.0)

    def _reflex_worker(self):
        """Low-latency reflexive thinking."""
        while self.running:
            try:
                ctx = reflex_queue.get(timeout=0.1)
                decision = self.agent.think_fast(ctx['step'], ctx['edge_stats'], ctx['loss_delta'])
                
                try:
                    reflex_decision_queue.put_nowait(decision)
                except queue.Full:
                    try:
                        reflex_decision_queue.get_nowait()
                        reflex_decision_queue.task_done()
                    except queue.Empty:
                        pass
                    reflex_decision_queue.put_nowait(decision)
                
                reflex_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"System 1 Error: {e}")

    def _strategic_worker(self):
        """High-latency strategic thinking."""
        while self.running:
            try:
                ctx = strategic_queue.get(timeout=0.1)
                decision = self.agent.think_slow(ctx['history'], ctx['regime_data'], ctx['model_state'])
                
                try:
                    strategic_decision_queue.put_nowait(decision)
                except queue.Full:
                    try:
                        strategic_decision_queue.get_nowait()
                        strategic_decision_queue.task_done()
                    except queue.Empty:
                        pass
                    strategic_decision_queue.put_nowait(decision)
                
                strategic_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"System 2 Error: {e}")

    @staticmethod
    def request_reflex(step: int, edge_stats: dict, loss_delta: float):
        if reflex_queue.full():
            try: reflex_queue.get_nowait()
            except queue.Empty: pass
        reflex_queue.put({'step': step, 'edge_stats': edge_stats, 'loss_delta': loss_delta})

    @staticmethod
    def request_strategic(history: dict, regime_data: dict, model_state: dict):
        if strategic_queue.full():
            try: strategic_queue.get_nowait()
            except queue.Empty: pass
        strategic_queue.put({'history': history, 'regime_data': regime_data, 'model_state': model_state})

    # ------------------------------------------------------------------
    # Mutation application (supports both APIs with Atomic Rollback)
    # ------------------------------------------------------------------

    def apply_pending_mutations(self, model: Any, optimizer: Any = None):
        """Processes decisions from both fast and slow thinking loops."""
        
        # 1. Process Reflexive (Fast) Decisions
        while not reflex_decision_queue.empty():
            try:
                decision = reflex_decision_queue.get_nowait()
                print(f"⚡ Reflexive Action: {decision.reasoning}")
                for edge_id in decision.prunes:
                    TopologicalMutator.mutate_edge(model, edge_id, "PRUNE")
                if optimizer and decision.lr_adjustment != 1.0:
                    for pg in optimizer.param_groups:
                        pg["lr"] *= decision.lr_adjustment
                    print(f"Reflex LR adjustment: x{decision.lr_adjustment:.4f}")
                reflex_decision_queue.task_done()
            except queue.Empty: break

        # 2. Process Strategic (Slow) / Legacy Decisions
        # Strategic and legacy use similar mutation logic, strategic adds atomic rollback
        
        # Helper to apply mutations with rollback
        def _apply_mutations_safe(mutations):
            if not mutations: return
            edge_snapshots = {}
            for mut in mutations:
                try:
                    l_idx, i_idx, o_idx = TopologicalMutator.parse_edge_id(mut.edge_id)
                    if l_idx < len(model.layers):
                        key = (l_idx, i_idx, o_idx)
                        edge_snapshots[key] = copy.deepcopy(model.layers[l_idx].edges[i_idx][o_idx])
                except Exception: pass
            try:
                for mut in mutations:
                    status = TopologicalMutator.mutate_edge(model, mut.edge_id, mut.action, mut.formula, mut.initial_params)
                    print(f"Mutation status: {status}")
            except Exception as e:
                print(f"Mutation failed: {e}. Rolling back {len(edge_snapshots)} edge(s).")
                for (l_idx, i_idx, o_idx), saved in edge_snapshots.items():
                    model.layers[l_idx].edges[i_idx][o_idx] = saved
                print("Rollback complete.")

        # Process Strategic
        while not strategic_decision_queue.empty():
            try:
                decision = strategic_decision_queue.get_nowait()
                if getattr(decision, "training_command", "CONTINUE") == "HALT":
                    print("🚨 Strategic HALT received!")
                    return "HALT"
                print(f"🧠 Strategic Review: {decision.reasoning}")
                _apply_mutations_safe(getattr(decision, "mutations", []))
                regime = getattr(decision, "regime_analysis", None)
                if regime and regime.hmm_transition_detected:
                    print(f"🚨 Regime Shift: {regime.predicted_regime}")
                strategic_decision_queue.task_done()
            except queue.Empty: break

        # Process Legacy
        while not decision_queue.empty():
            try:
                decision = decision_queue.get_nowait()
                if decision.training_command == "HALT": return "HALT"
                print(f"Applying legacy mutations. Reasoning: {decision.reasoning}")
                _apply_mutations_safe(decision.mutations)
                decision_queue.task_done()
            except queue.Empty: break
            
        return "CONTINUE"
