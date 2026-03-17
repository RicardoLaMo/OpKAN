from pydantic import BaseModel, Field
from typing import List, Optional

class EdgeMutation(BaseModel):
    """
    Defines a mutation for a specific KAN edge.
    """
    layer_idx: int = Field(..., description="Index of the KAN layer.")
    input_idx: int = Field(..., description="Index of the input neuron.")
    output_idx: int = Field(..., description="Index of the output neuron.")
    symbolic_expression: str = Field(..., description="The Python/Torch symbolic expression (e.g., 'torch.pow(x, 2)'). Must be C^2 continuous.")
    explanation: str = Field(..., description="Brief explanation of why this mutation was chosen.")

class LiuClawDecision(BaseModel):
    """
    The structured decision output from the LiuClaw agent.
    """
    reasoning: str = Field(..., description="Chain-of-thought reasoning for the current set of mutations.")
    mutations: List[EdgeMutation] = Field(default_factory=list, description="List of edge mutations to apply.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the proposed mutations.")
    regime_adjustment: Optional[str] = Field(None, description="Instructions for adjusting training priors based on detected market regime.")
