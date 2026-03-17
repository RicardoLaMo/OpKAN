import random
import time
import json
import threading
from datetime import datetime
from typing import List, Optional, Literal

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, RichLog, Label
from textual.reactive import reactive
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from textual_plotext import PlotextPlot
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Models for Scenario Generation
# ---------------------------------------------------------------------------

class ScenarioStep(BaseModel):
    regime: str = Field(..., description="Target regime name (e.g. VOL_SPIKE, CRASH, RECOVERY)")
    loss_trend: Literal["UP", "DOWN", "FLAT"] = Field(..., description="Direction of PDE loss")
    agent_reasoning: str = Field(..., description="Narrative reasoning from the agent")
    mutation_action: Optional[str] = Field(None, description="e.g. REPLACE Edge L0_N5")
    mutation_formula: Optional[str] = Field(None, description="e.g. torch.exp(x)")

class ScenarioPlan(BaseModel):
    name: str
    steps: List[ScenarioStep]

# ---------------------------------------------------------------------------
# TUI Components
# ---------------------------------------------------------------------------

class StoryPanel(Static):
    """The static/narrative sidebar describing the OpKAN project with Unicode Math."""
    def on_mount(self) -> None:
        self.update_story()

    def update_story(self):
        story_md = """
# 🛡️ OpKAN Deep-Dive
**Physics-Informed KAN for Options**

### 1. The Math (Heston PDE)
∂V/∂t + ½vS² ∂²V/∂S² + ρσvS ∂²V/∂S∂v + 
½σ²v ∂²V/∂v² + rS ∂V/∂S + κ(θ-v) ∂V/∂v - rV = 0

### 2. KAN vs MLP
- **B-splines (C² smooth)**: Essential for 2nd order residuals.
- **Symbolic Mutation**: LiuClaw discovers physics-aligned formulas.

### 3. Dual-Process Brain
- **System 1**: Reflexive pruning (Fast).
- **System 2**: Strategic Review (Slow).
"""
        self.update(Markdown(story_md))

class MetricCard(Static):
    """A small card showing a single metric with business context."""
    value = reactive("0")
    
    def __init__(self, title: str, unit: str = "", subtitle: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.unit = unit
        self.subtitle = subtitle

    def render(self) -> Panel:
        return Panel(
            f"[bold cyan]{self.value}[/] {self.unit}\n[dim]{self.subtitle}[/]",
            title=self.title,
            border_style="green"
        )

class BrainStatus(Static):
    """Visualizes the active state of System 1 and System 2."""
    s1_active = reactive(False)
    s2_active = reactive(False)
    dual_mode = reactive(True)

    def render(self) -> Panel:
        s1_style = "bold green" if self.s1_active else "dim white"
        s2_style = "bold yellow" if self.s2_active else "dim white"
        mode_str = "[bold magenta]DUAL-BRAIN ENABLED[/]" if self.dual_mode else "[bold red]BASELINE ONLY[/]"
        
        table = Table.grid(expand=True)
        table.add_column()
        table.add_row(Text(mode_str))
        table.add_row(Text("● System 1 (Reflex)", style=s1_style))
        table.add_row(Text("● System 2 (Strategic)", style=s2_style))
        
        return Panel(table, title="Neural Engine", border_style="blue")

class OpKANDashboard(App):
    """The main OpKAN Terminal Telemetry App."""
    
    CSS = """
    Screen {
        background: #0a0a0a;
    }
    #sidebar {
        width: 40;
        background: #111;
        padding: 1;
        border-right: double green;
    }
    #main-content {
        padding: 1;
    }
    #metrics-row {
        height: 7;
    }
    #plots-row {
        height: 1fr;
    }
    #log-row {
        height: 12;
        border-top: dashed #333;
    }
    #brain-panel {
        height: 7;
        margin-top: 1;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dual", "Toggle Dual Brain"),
        ("s", "new_scenario", "LLM Generate Scenario"),
    ]

    throughput = reactive(0)
    pde_loss = reactive(1.0)
    regime = reactive("INITIALIZING")
    dual_mode = reactive(True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield StoryPanel()
                yield BrainStatus(id="brain-panel")
            with Vertical(id="main-content"):
                with Horizontal(id="metrics-row"):
                    yield MetricCard(title="Pricing Velocity", unit="opts/s", subtitle="H200 Inference Speed", id="card-tput")
                    yield MetricCard(title="PDE Error (MSE)", subtitle="Consistency Gap", id="card-loss")
                    yield MetricCard(title="Market Context", subtitle="LLM-Decoded Regime", id="card-regime")
                
                with Horizontal(id="plots-row"):
                    yield PlotextPlot(id="loss-plot")
                    yield PlotextPlot(id="tput-plot")
                
                with Vertical(id="log-row"):
                    yield Label("[bold yellow]🧠 LiuClaw reasoning & Market Events[/]")
                    yield RichLog(id="agent-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.loss_history = []
        self.tput_history = []
        self.step_count = 0
        self.current_plan: Optional[ScenarioPlan] = None
        self.current_step_idx = 0
        
        self.init_plots()
        self.query_one("#agent-log").write("[bold green]System Initialized. H200 GPU Online.[/]")
        self.action_new_scenario() # Trigger first LLM scenario

    def init_plots(self):
        self.query_one("#loss-plot").plt.theme("dark")
        self.query_one("#loss-plot").plt.title("Heston PDE Residual (Live)")
        self.query_one("#tput-plot").plt.theme("dark")
        self.query_one("#tput-plot").plt.title("Compute Throughput")
        self.set_interval(0.5, self.update_data)

    def action_toggle_dual(self):
        self.dual_mode = not self.dual_mode
        self.query_one("#brain-panel").dual_mode = self.dual_mode
        status = "ENABLED" if self.dual_mode else "DISABLED"
        self.query_one("#agent-log").write(f"[bold magenta]Brain Overwrite: Dual-Process Engine {status}.[/]")

    def action_new_scenario(self):
        """Simulates a call to vLLM to generate a complex market scenario."""
        self.query_one("#agent-log").write("[dim italic]Querying vLLM for new market scenario...[/]")
        # In a real environment, this calls InstructorClient
        # For the demo, we simulate the LLM's imagination
        scenarios = [
            ScenarioPlan(name="Flash Crash", steps=[
                ScenarioStep(regime="STABLE", loss_trend="DOWN", agent_reasoning="Normal diffusion detected."),
                ScenarioStep(regime="STRESS", loss_trend="UP", agent_reasoning="LIQUIDITY GAP! S&P -3% in 50ms. Curvature snapping."),
                ScenarioStep(regime="RECOVERY", loss_trend="DOWN", agent_reasoning="Stabilizing via Gamma-Scalping nodes.")
            ]),
            ScenarioStep(regime="Vol Expansion", loss_trend="UP", agent_reasoning="CPI data hotter than expected. Skew steepening.", mutation_action="REPLACE L0_N0", mutation_formula="torch.exp(x)")
        ]
        # We'll actually just randomize one for the demo
        names = ["FOMC Hawk Pivot", "Geopolitical Supply Shock", "Earnings Momentum Run"]
        regimes = ["STABLE", "EXPANSION", "CRASH", "JUMP_DIFFUSION"]
        
        new_steps = []
        for _ in range(5):
            new_steps.append(ScenarioStep(
                regime=random.choice(regimes),
                loss_trend=random.choice(["UP", "DOWN", "FLAT"]),
                agent_reasoning=f"LLM Reasoning: Stochastic volatility showing {random.choice(['heavy tails', 'mean reversion', 'non-linear clusters'])}."
            ))
        
        self.current_plan = ScenarioPlan(name=random.choice(names), steps=new_steps)
        self.current_step_idx = 0
        self.query_one("#agent-log").write(f"[bold cyan]vLLM Scenario Active:[/] {self.current_plan.name}")

    def update_data(self) -> None:
        self.step_count += 1
        
        # 1. Consume Scenario Plan
        if self.current_plan:
            step = self.current_plan.steps[self.current_step_idx]
            self.regime = step.regime
            
            # Loss trend logic
            if step.loss_trend == "UP":
                self.pde_loss *= (1.05 + random.random()*0.05)
            elif step.loss_trend == "DOWN":
                # System 2 helps more in dual mode
                decay = 0.95 if self.dual_mode else 0.98
                self.pde_loss = max(self.pde_loss * decay, 0.0001)
            else:
                self.pde_loss += (random.random() - 0.5) * 0.01
            
            # Periodically advance scenario step
            if self.step_count % 10 == 0:
                self.current_step_idx = (self.current_step_idx + 1) % len(self.current_plan.steps)
                self.query_one("#agent-log").write(f"[dim italic]Event Update:[/] {step.agent_reasoning}")
        
        # Clamp loss
        self.pde_loss = max(min(self.pde_loss, 50.0), 0.0001)
        self.throughput = 26300 + random.randint(-200, 200)
        
        self.loss_history.append(self.pde_loss)
        self.tput_history.append(self.throughput)
        if len(self.loss_history) > 60:
            self.loss_history.pop(0)
            self.tput_history.pop(0)
            
        # 2. Update UI
        self.query_one("#card-tput").value = f"{self.throughput:,}"
        self.query_one("#card-loss").value = f"{self.pde_loss:.6f}"
        self.query_one("#card-regime").value = self.regime
        
        self.refresh_plots()
        
        # 3. Brain Activity
        if self.dual_mode:
            brain = self.query_one("#brain-panel")
            brain.s1_active = (self.step_count % 8 == 0)
            brain.s2_active = (self.step_count % 25 == 0)
            
            if brain.s1_active:
                self.query_one("#agent-log").write(f"[dim]System 1:[/] PRUNE dead edge L0_N{random.randint(0,15)}.")
            if brain.s2_active:
                self.query_one("#agent-log").write(f"[bold cyan]System 2:[/] Strategic review of {self.regime} completed.")

    def refresh_plots(self):
        l_p = self.query_one("#loss-plot")
        l_p.plt.clear_data()
        l_p.plt.plot(self.loss_history, color="red")
        l_p.refresh()
        
        t_p = self.query_one("#tput-plot")
        t_p.plt.clear_data()
        t_p.plt.plot(self.tput_history, color="green")
        t_p.refresh()

if __name__ == "__main__":
    app = OpKANDashboard()
    app.run()
