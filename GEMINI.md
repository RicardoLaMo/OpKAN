# Project: Physics-Informed KAN (PI-KAN) for Options Pricing & Regime Detection

## Architecture Overview
1. **The Math:** A PyTorch KAN solving the Heston PDE.
2. **The Data:** OPRA options data converted to (S, v, t) tensors.
3. **The Agent:** An LLM that uses chain-of-thought to mutate the KAN's B-spline edges into strictly C^2 continuous symbolic functions (e.g., `torch.pow(x, 2)`).
4. **The Output:** Extracted structural features fed into a Gaussian HMM for regime switching.

## Agent Instructions
When writing code for this repository, strictly adhere to:
- PyTorch autograd best practices (always use `create_graph=True` for high-order derivatives).
- Asynchronous data loading (`pin_memory=True`, `non_blocking=True`).
- Pydantic for any LLM structured outputs.
