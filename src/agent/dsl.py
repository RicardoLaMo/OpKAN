from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class EdgeMutation(BaseModel):
    """
    Defines a mutation for a specific KAN edge with tool-calling schema.
    """
    edge_id: str = Field(..., description="The exact ID of the B-spline edge, e.g., 'L0_N0_to_L1_N1'")
    action: Literal["PRUNE", "REPLACE", "KEEP"] = Field(..., description="The topological action to take on the edge.")
    formula: Optional[str] = Field(None, description="If action is REPLACE, provide the strictly C^2 continuous PyTorch formula (e.g., 'torch.pow(x, 2)').")
    initial_params: Optional[dict] = Field(None, description="Optional initial values for symbolic parameters like scale, shift, or coefficients.")
    reasoning: str = Field(..., description="A rigorous quantitative justification for this action based on L1 norms and sampled points.")

class RegimeThesis(BaseModel):
    """
    The structural thesis explaining the current market regime shift.
    """
    hmm_transition_detected: bool = Field(..., description="True if the HMM probabilities indicate an impending regime shift.")
    predicted_regime: Optional[int] = Field(None, description="The target regime state (0: Diffusion, 1: Vol Expansion, 2: Jump/Crash)")
    thesis_statement: Optional[str] = Field(None, description="A 3-sentence thesis explaining what structural changes in the volatility surface (based on the KAN) triggered this shift.")

# ---------------------------------------------------------------------------
# Dual-process (System 1 / System 2) decision types
# ---------------------------------------------------------------------------

class ReflexDecision(BaseModel):
    """
    Fast (System 1) decision produced by think_fast().
    Handles immediate edge pruning and learning-rate adjustments.
    """
    reasoning: str = Field(..., description="Fast justification for reflexive maintenance.")
    prunes: List[str] = Field(default_factory=list, description="List of edge IDs to immediately PRUNE.")
    lr_adjustment: float = Field(1.0, description="Multiplier for the current learning rate (e.g., 0.9 to decay, 1.1 to boost).")

class StrategicDecision(BaseModel):
    """
    Slow (System 2) decision produced by think_slow().
    Handles deliberate topological mutations and regime analysis.
    """
    reasoning: str = Field(..., description="Chain-of-thought reasoning over long-term trends and mathematical stability.")
    mutations: List[EdgeMutation] = Field(default_factory=list, description="List of complex topological mutations (REPLACE).")
    regime_analysis: RegimeThesis = Field(..., description="Deep analysis of market regime shifts.")
    training_command: Literal["CONTINUE", "HALT"] = Field("CONTINUE", description="Emergency halt command.")

class LiuClawDecision(BaseModel):
    """
    Master payload received by the PyTorch training loop.
    Guaranteed valid JSON schema through instructor/vLLM.
    Supports backward compatibility and dual-process responses.
    """
    training_command: Literal["CONTINUE", "HALT"] = Field("CONTINUE", description="HALT only if arbitrage violations or severe data corruption is detected.")
    reasoning: Optional[str] = Field(None, description="Overall reasoning for the proposed mutations and regime analysis.")
    mutations: List[EdgeMutation] = Field(default_factory=list, description="List of topological optimizations for the KAN.")
    regime_analysis: Optional[RegimeThesis] = Field(None, description="The agent's analysis of market regime shifts.")
    strategic: Optional[StrategicDecision] = None
    reflex: Optional[ReflexDecision] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for the overall decision.")


class ReflexDecision(BaseModel):
    """
    Fast-path (think_fast) response for low-latency pruning and LR adjustments.
    """
    reasoning: str = Field(..., description="Brief justification for the reflex action.")
    prunes: List[str] = Field(default_factory=list, description="List of edge IDs to prune immediately.")
    lr_adjustment: float = Field(default=1.0, description="Multiplicative factor to apply to the current learning rate.")


class StrategicDecision(BaseModel):
    """
    Slow-path (think_slow) response for full topological mutation and regime analysis.
    """
    reasoning: str = Field(..., description="Overall reasoning for the proposed mutations and regime analysis.")
    mutations: List[EdgeMutation] = Field(default_factory=list, description="List of topological optimizations for the KAN.")
    regime_analysis: RegimeThesis = Field(..., description="The agent's analysis of market regime shifts.")
    training_command: Literal["CONTINUE", "HALT"] = Field(default="CONTINUE", description="HALT only if arbitrage violations or severe data corruption is detected.")
