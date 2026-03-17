# PRD: Dual-Process Thinking Architecture for OpKAN

## 1. Objective
Enhance the LiuClaw agent by implementing a "Fast and Slow Thinking" (System 1 & System 2) architecture. The goal is to balance near-instantaneous topological maintenance with deep, mathematically rigorous structural mutations to optimize the PI-KAN solver for the Heston PDE.

## 2. Background (Dual-Process Theory)
- **System 1 (Fast Thinking)**: Automatic, frequent, low-latency. Handles "reflexive" tasks like pruning near-zero weights or adjusting learning rate schedules based on immediate loss spikes.
- **System 2 (Slow Thinking)**: Effortful, infrequent, high-latency. Performs deep reasoning over curvature stability, HMM regime transitions, and symbolic expression selection.

## 3. Architecture Overview

### A. The Math Loop (Execution)
- Continues at >20k samples/sec on H200.
- Queries System 1 every **50 epochs**.
- Queries System 2 every **500 epochs**.

### B. System 1: The "Heuristic Brain"
- **Model**: Smaller, high-speed LLM (e.g., Qwen 2.5 7B Coder) or a specialized "reflex" prompt for the 32B model.
- **Input**: Immediate stats (current loss, mean edge activation).
- **Output**: PRUNE commands, KEEP commands, learning rate tweaks.
- **Latency Target**: < 50ms.

### C. System 2: The "Strategic Brain"
- **Model**: Full-reasoning model (e.g., DeepSeek-R1 or Qwen 2.5 72B Instruct).
- **Input**: Rich historical context (Loss trajectory, HMM regime probabilities, second-order gradient stability maps).
- **Output**: REPLACE commands (symbolic mutations), regime-specific prior adjustments.
- **Latency Target**: 1s - 5s.

## 4. Performance Objectives
1.  **Convergence Speed**: Reduce epochs to target loss by 20% through "reflexive" pruning.
2.  **Mathematical Rigor**: Achieve zero violations of financial monotonicity by using System 2 to select physics-aligned symbolic functions.
3.  **Compute Efficiency**: Minimize H200 idle time by offloading reflexive tasks to System 1.

## 5. Integration Plan
1.  **Refined DSL**: Split `LiuClawDecision` into `ReflexDecision` (System 1) and `StrategicDecision` (System 2).
2.  **Async Coordinator Update**: Implement two separate worker threads or a tiered queue to handle concurrent fast/slow requests.
3.  **Comparison Framework**: Build a benchmarking script to compare:
    - **Baseline**: No LLM (Pure Gradient Descent).
    - **Single-Brain**: Current architecture (One LLM call for all).
    - **Dual-Brain**: The new Fast/Slow architecture.

## 6. Document Update
The results and the theoretical justification for Dual-Process Thinking in quantitative physics will be added to the LaTeX paper.
