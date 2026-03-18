"""
Rule-based fallback agent used when the vLLM server is unreachable.

Mirrors the think_fast / think_slow interface of LiuClawAgent but makes
decisions purely from edge statistics and HMM data — no LLM calls.

System 1 (think_fast): prune edges below L1 threshold 0.005; decay LR on rising loss.
System 2 (think_slow): prune deeply-dormant edges; detect regime shifts from HMM
                       transition matrix off-diagonal mass.
"""
import numpy as np
from typing import Dict, Any

from src.agent.dsl import ReflexDecision, StrategicDecision, EdgeMutation, RegimeThesis

# Paper thresholds (§4.3)
L1_S1_PRUNE_THRESHOLD = 0.005   # System 1: stagnant B-spline edges
L1_S2_PRUNE_THRESHOLD = 0.001   # System 2: deeply dormant edges
HMM_TRANSITION_MASS   = 0.30    # off-diagonal sum above this → transition detected

REGIME_LABELS = {0: "Diffusion/Stable", 1: "Vol Expansion", 2: "Jump/Crash"}


class RuleBasedFallbackAgent:
    """
    Deterministic rule-based agent implementing the dual-process interface.
    Instantiated automatically when the vLLM endpoint is unreachable.
    """

    def think_fast(
        self,
        step: int,
        edge_stats: Dict[str, Any],
        loss_delta: float,
    ) -> ReflexDecision:
        """
        System 1 (reflexive, <1 ms): prune stagnant edges; adjust learning rate.

        Args:
            step:       current training step
            edge_stats: {edge_id: {"l1_norm": float, "type": str}, ...}
            loss_delta: loss.item() at this step (positive = loss rose, negative = fell)
        """
        prunes = [
            edge_id
            for edge_id, stats in edge_stats.items()
            if isinstance(stats, dict) and stats.get("l1_norm", 1.0) < L1_S1_PRUNE_THRESHOLD
        ]
        lr_adjustment = 0.9 if loss_delta > 0.001 else 1.0

        reasoning = (
            f"Rule-based S1 at step {step}: {len(prunes)} edges below "
            f"L1={L1_S1_PRUNE_THRESHOLD:.3f} queued for pruning; "
            f"lr_adjustment={lr_adjustment:.1f} (loss_delta={loss_delta:.6f})."
        )
        return ReflexDecision(
            reasoning=reasoning,
            prunes=prunes,
            lr_adjustment=lr_adjustment,
        )

    def think_slow(
        self,
        history: Dict[str, Any],
        regime_data: Dict[str, Any],
        model_state: Dict[str, Any],
    ) -> StrategicDecision:
        """
        System 2 (strategic, ~0 ms): emit PRUNE mutations for deeply dormant
        edges; use HMM transition matrix to assess regime shift probability.

        Args:
            history:      {"step": int, "loss": float, "current_regime_id": int}
            regime_data:  {"transition_matrix": list[list], "means": list, "covars": list}
                          or {} before the HMM has been fitted
            model_state:  {edge_id: {"l1_norm": float, "type": str}, ...}
        """
        mutations = []
        for edge_id, stats in model_state.items():
            if not isinstance(stats, dict):
                continue
            if (
                stats.get("type") == "bspline"
                and stats.get("l1_norm", 1.0) < L1_S2_PRUNE_THRESHOLD
            ):
                mutations.append(
                    EdgeMutation(
                        edge_id=edge_id,
                        action="PRUNE",
                        formula=None,
                        reasoning=(
                            f"L1 norm {stats['l1_norm']:.6f} below strategic "
                            f"prune threshold {L1_S2_PRUNE_THRESHOLD}."
                        ),
                    )
                )

        # HMM-based regime detection
        transmat = regime_data.get("transition_matrix")
        predicted_regime = int(history.get("current_regime_id", 0))
        hmm_transition_detected = False
        off_diag_str = "N/A (HMM not yet fitted)"

        if transmat is not None:
            arr = np.array(transmat)
            off_diag_sum = float(arr.sum() - np.trace(arr))
            hmm_transition_detected = off_diag_sum > HMM_TRANSITION_MASS
            off_diag_str = f"{off_diag_sum:.3f}"

        step = history.get("step", 0)
        regime_name = REGIME_LABELS.get(predicted_regime, str(predicted_regime))
        thesis = (
            f"Rule-based S2 at step {step}: {len(mutations)} structural prune mutation(s) "
            f"emitted. HMM off-diagonal mass={off_diag_str}, "
            f"transition_detected={hmm_transition_detected}, "
            f"current_regime={regime_name}."
        )
        return StrategicDecision(
            reasoning=thesis,
            mutations=mutations,
            regime_analysis=RegimeThesis(
                hmm_transition_detected=hmm_transition_detected,
                predicted_regime=predicted_regime,
                thesis_statement=thesis,
            ),
            training_command="CONTINUE",
        )
