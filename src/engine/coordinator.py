import copy
import threading
import time
import queue
from typing import Dict, Any, Callable
from src.agent.core import LiuClawAgent
from src.models.mutator import TopologicalMutator
from src.engine.queues import context_queue, decision_queue

class EngineCoordinator:
    """
    Coordinates the async interaction between the PyTorch KAN training loop
    and the LiuClaw agent.
    """
    def __init__(self, agent: LiuClawAgent):
        self.agent = agent
        self.running = False
        self.agent_thread = None

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
                # Wait for new context (non-blocking with timeout to allow exit)
                context = context_queue.get(timeout=0.1)
                
                # Unpack context (kan_state, pipeline_health)
                kan_state, pipeline_health = context
                
                # Perform reasoning (slow call to LLM)
                decision = self.agent.decide_mutations(kan_state, pipeline_health)
                
                # Push decision to the queue; drop oldest if full to avoid blocking worker
                try:
                    decision_queue.put_nowait(decision)
                except queue.Full:
                    print(f"Decision queue full (maxsize={decision_queue.maxsize}); "
                          "dropping stale decision to make room.")
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
    def apply_pending_mutations(model: Any):
        """Checks the decision queue and applies any pending mutations."""
        while not decision_queue.empty():
            try:
                decision = decision_queue.get_nowait()
                
                if decision.training_command == "HALT":
                    print("🚨 LiuClaw HALT command received! Terminating training loop.")
                    return "HALT"

                print(f"Applying mutations from agent. Reasoning: {decision.reasoning}")

                if decision.mutations:
                    # Snapshot affected edge modules for rollback
                    edge_snapshots = {}
                    for mutation in decision.mutations:
                        try:
                            layer_idx, in_idx, out_idx = TopologicalMutator.parse_edge_id(mutation.edge_id)
                            if layer_idx < len(model.layers):
                                key = (layer_idx, in_idx, out_idx)
                                edge_snapshots[key] = copy.deepcopy(
                                    model.layers[layer_idx].edges[in_idx][out_idx]
                                )
                        except (ValueError, AttributeError):
                            pass
                    try:
                        for mutation in decision.mutations:
                            status = TopologicalMutator.mutate_edge(
                                model,
                                mutation.edge_id,
                                mutation.action,
                                mutation.formula,
                                mutation.initial_params
                            )
                            print(f"Mutation status: {status}")
                    except Exception as e:
                        print(f"Mutation failed: {e}. Rolling back {len(edge_snapshots)} edge(s).")
                        for (layer_idx, in_idx, out_idx), saved_module in edge_snapshots.items():
                            model.layers[layer_idx].edges[in_idx][out_idx] = saved_module
                        print("Rollback complete.")

                if decision.regime_analysis.hmm_transition_detected:
                    print(f"Regime Shift Detected: {decision.regime_analysis.predicted_regime}!")
                    print(f"Thesis: {decision.regime_analysis.thesis_statement}")

                decision_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Failed to apply mutation: {e}")
        return "CONTINUE"

    @staticmethod
    def request_mutation(kan_state: Dict[str, Any], pipeline_health: Dict[str, Any]):
        """Submits context to the agent for decision making."""
        if context_queue.full():
            try:
                context_queue.get_nowait()
            except queue.Empty:
                pass
        
        context_queue.put((kan_state, pipeline_health))
