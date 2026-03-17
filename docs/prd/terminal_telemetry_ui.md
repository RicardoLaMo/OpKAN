# PRD: OpKAN Terminal Storyboard & Telemetry (TUI)

## 1. Objective
Create a high-fidelity Terminal User Interface (TUI) to visualize the OpKAN project story, mathematical foundation, and real-time H200 performance. This serves as a "terminal-native" alternative to Streamlit, optimized for SSH-based remote management.

## 2. Target Audience
- Quantitative researchers monitoring H200 training runs.
- Developers auditing the LiuClaw agent's "Chain of Thought."
- Stakeholders reviewing the OpKAN architectural "story."

## 3. Key Components (TUI Layout)

### A. The "Story" Sidebar (Static/Narrative)
- **Math**: Rendered LaTeX-like text for the Heston PDE and KAN theorem.
- **Architecture**: Text-based diagrams of the four-tier hierarchy.
- **dual-process**: Explanation of System 1 (Reflex) and System 2 (Strategic).

### B. Live Telemetry (Dynamic)
- **HMM Dashboard**: Real-time regime label (Diffusion, Expansion, Crash).
- **Math Monitor**: Live line charts for PDE and Boundary Condition loss.
- **Agent Feed**: A live-scrolling terminal showing LiuClaw's structured Pydantic decisions and justifications.
- **H200 Stats**: Samples/sec throughput and GPU status.

## 4. Tech Stack
- **Framework**: `Textual` (for modern TUI widgets and layout).
- **Rendering**: `Rich` (for tables, syntax highlighting, and formatting).
- **Plotting**: `textual-plotext` (for terminal-based line charts).

## 5. Integration
- Connects to the existing `EngineCoordinator` via a shared `state` object or a lightweight IPC mechanism.
- Runs as a standalone script: `python3 src/ui/tui/app.py`.

## 6. Success Criteria
- Zero blocking of the PyTorch math loop.
- Refresh rate of at least 2Hz.
- Visually "impressive" terminal experience using H200's processing power.
