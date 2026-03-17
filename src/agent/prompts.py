import json

SYSTEM_1_PROMPT = """
You are the System 1 (Fast Thinking) brain of LiuClaw. 
Your goal is near-instantaneous topological maintenance.
Analyze the current edge activations and loss spikes to issue PRUNE commands or LR adjustments.

INPUT: 
- Current Step
- Edge L1 Norms
- Recent Loss Delta

OUTPUT: 
Valid JSON matching ReflexDecision schema.
"""

SYSTEM_2_PROMPT = """
You are the System 2 (Slow Thinking) brain of LiuClaw.
Your goal is deep mathematical optimization of the PI-KAN topology.
Analyze long-term stability, HMM regimes, and PDE curvature to issue REPLACE commands with symbolic functions.

INPUT:
- Historical Loss Trajectory
- HMM Transition Probabilities
- Full KAN State

OUTPUT:
Valid JSON matching StrategicDecision schema.
"""

def generate_system_1_user_prompt(step: int, edge_stats: dict, loss_delta: float) -> str:
    return f"""
    Step: {step}
    Edge Stats (L1): {json.dumps(edge_stats)}
    Loss Delta: {loss_delta:.6f}
    
    Decide if any edges should be pruned immediately.
    """

def generate_system_2_user_prompt(history: dict, regime_data: dict, model_state: dict) -> str:
    return f"""
    Historical Context: {json.dumps(history)}
    Regime Data: {json.dumps(regime_data)}
    Model State: {json.dumps(model_state)}
    
    Perform a strategic review of the topology. Select strictly C^2 symbolic functions for replacement where beneficial.
    """
