from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class EdgeMutation(BaseModel):
    """
    Defines a mutation for a specific KAN edge with tool-calling schema.
    """
    edge_id: str = Field(..., description="The exact ID of the B-spline edge, e.g., 'L0_N0_to_L1_N1'")
    action: Literal["PRUNE", "REPLACE", "KEEP"] = Field(..., description="The topological action to take on the edge.")
    formula: Optional[str] = Field(None, description="If action is REPLACE, provide the strictly C^2 continuous PyTorch formula (e.g., 'torch.pow(x, 2)'). Otherwise null.")
    reasoning: str = Field(..., description="A rigorous quantitative justification for this action based on L1 norms and sampled points.")

class RegimeThesis(BaseModel):
    """
    The structural thesis explaining the current market regime shift.
    """
    hmm_transition_detected: bool = Field(..., description="True if the HMM probabilities indicate an impending regime shift.")
    predicted_regime: Optional[int] = Field(None, description="The target regime state (0: Diffusion, 1: Vol Expansion, 2: Jump/Crash)")
    thesis_statement: Optional[str] = Field(None, description="A 3-sentence thesis explaining what structural changes in the volatility surface (based on the KAN) triggered this shift.")

class LiuClawDecision(BaseModel):
    """
    The master payload received by the PyTorch training loop.
    Guaranteed valid JSON schema through instructor/vLLM.
    """
    training_command: Literal["CONTINUE", "HALT"] = Field(..., description="HALT only if arbitrage violations or severe data corruption is detected.")
    reasoning: str = Field(..., description="Overall reasoning for the proposed mutations and regime analysis.")
    mutations: List[EdgeMutation] = Field(default_factory=list, description="List of topological optimizations for the KAN.")
    regime_analysis: RegimeThesis = Field(..., description="The agent's analysis of market regime shifts.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for the overall decision.")
