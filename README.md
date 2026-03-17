# OpKAN: Physics-Informed Kolmogorov-Arnold Networks for Options Pricing

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![H200 Optimized](https://img.shields.io/badge/Compute-H200_Optimized-green.svg)

OpKAN is a high-performance quantitative finance engine designed to solve the **Heston Stochastic Volatility PDE** using **Physics-Informed Kolmogorov-Arnold Networks (PI-KAN)**. Powered by the **NVIDIA H200 GPU**, OpKAN integrates a real-time reasoning agent (**LiuClaw**) to surgically mutate network topology based on market regime detection.

## 🚀 Key Features

*   **Heston PDE Solver**: A PyTorch-based PI-KAN that enforces $C^2$ continuous boundary conditions and mathematical consistency through high-order Autograd residuals.
*   **LiuClaw reasoning Agent**: An LLM-driven "brain" (compatible with Qwen 2.5, DeepSeek-R1) that performs topological mutations (PRUNE, REPLACE) on the KAN in real-time.
*   **H200 Optimized**: Built for massive throughput using `vLLM` for fast reasoning and asynchronous PyTorch DataLoaders for math execution.
*   **Regime-Aware HMM**: A Gaussian Hidden Markov Model that extracts structural features from OPRA data to guide the agent's thesis.
*   **Dynamic Topology**: Evolves from flexible B-splines into strictly $C^2$ symbolic functions (e.g., `torch.pow(x, 2)`, `torch.exp(x)`) as the model learns.
*   **Telemetry Dashboard**: A live Streamlit suite for monitoring PDE convergence, agent chain-of-thought, and structural mutations.

## 🏗️ Architecture

1.  **The Math**: Solving $\frac{\partial V}{\partial t} + \frac{1}{2}vS^2\frac{\partial^2 V}{\partial S^2} + \rho\sigma vS\frac{\partial^2 V}{\partial S\partial v} + \frac{1}{2}\sigma^2v\frac{\partial^2 V}{\partial v^2} + rS\frac{\partial V}{\partial S} + \kappa(\theta-v)\frac{\partial V}{\partial v} - rV = 0$.
2.  **The Agent**: Uses a Pydantic-enforced DSL via `instructor` to guarantee structured decisions from the LLM.
3.  **The Mutator**: A "PyTorch Surgeon" that swaps active B-spline edges for symbolic math without breaking the computational graph.
4.  **The Data**: High-fidelity OPRA/Databento pipeline with real-time IV solving and spline surface fitting.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/RicardoLaMo/OpKAN.git
cd OpKAN

# Install dependencies
pip install torch pandas numpy scipy streamlit plotly hmmlearn instructor openai transformers peft
```

## 📈 Usage

### 1. Start the vLLM reasoning Server (H200)
```bash
./scripts/start_vllm.sh
```

### 2. Run Live Training Session
```bash
PYTHONPATH=. python3 scripts/live_session.py
```

### 3. Launch Telemetry Dashboards
OpKAN provides two ways to monitor the engine:

*   **Terminal Telemetry (TUI)**: Optimized for SSH/H200 environments.
    ```bash
    python3 src/ui/tui/app.py
    ```
*   **Streamlit (Browser)**: Rich visualization for local analysis.
    ```bash
    streamlit run src/ui/dashboard.py
    ```

### 4. Backtesting
```bash
PYTHONPATH=. python3 scripts/backtest_pikan.py
```

## 🧬 LLM Fine-Tuning (LoRA)

OpKAN supports LoRA fine-tuning to optimize the LiuClaw agent's mutation logic based on historical PDE convergence rewards.

```bash
# Collect training trajectories during live sessions
# Run the fine-tuning script
python3 scripts/fine_tune_liuclaw.py
```

## 📄 License
MIT License. Created by [RicardoLaMo](https://github.com/RicardoLaMo).
