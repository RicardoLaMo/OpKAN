import random
import time
import json
from datetime import datetime
from typing import List, Optional, Literal

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, Static, RichLog, Label, Button
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
    regime: str = Field(..., description="Target regime name")
    loss_trend: Literal["UP", "DOWN", "FLAT"] = Field(..., description="Direction of PDE loss")
    price_trend: Literal["UP", "DOWN", "FLAT"] = Field(..., description="Direction of Option Price")
    agent_reasoning: str = Field(..., description="Narrative reasoning from the agent")

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
        background: #050505;
    }
    #sidebar {
        width: 40;
        background: #0a0a0a;
        padding: 1;
        border-right: double green;
        transition: width 200ms;
    }
    #sidebar.hidden {
        width: 0;
        display: none;
    }
    #main-content {
        padding: 1;
    }
    #metrics-grid {
        height: 14;
        grid-size: 4 2;
        grid-gutter: 1;
    }
    #plots-grid {
        height: 1fr;
        grid-size: 2 2;
        grid-gutter: 1;
    }
    #log-row {
        height: 10;
        border-top: dashed #333;
    }
    .plot-container {
        border: solid #222;
    }
    #brain-panel {
        height: 7;
        margin-top: 1;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dual", "Toggle Dual Brain"),
        ("s", "new_scenario", "New Scenario"),
        ("b", "toggle_sidebar", "Toggle Sidebar"),
    ]

    throughput = reactive(0)
    pde_loss = reactive(1.0)
    option_price = reactive(10.0)
    delta = reactive(0.5)
    gamma = reactive(0.02)
    vega = reactive(0.1)
    regime = reactive("INITIALIZING")
    dual_mode = reactive(True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield StoryPanel()
                yield BrainStatus(id="brain-panel")
            with Vertical(id="main-content"):
                with Grid(id="metrics-grid"):
                    yield MetricCard(title="Pricing Velocity", unit="opts/s", subtitle="H200 Inference", id="card-tput")
                    yield MetricCard(title="PDE Error (MSE)", subtitle="Physics Gap", id="card-loss")
                    yield MetricCard(title="Option Price", unit="$", subtitle="Market Value", id="card-price")
                    yield MetricCard(title="Market Regime", subtitle="LLM Decoded", id="card-regime")
                    yield MetricCard(title="Delta (Δ)", subtitle="Price Sensitivity", id="card-delta")
                    yield MetricCard(title="Gamma (Γ)", subtitle="Delta Sensitivity", id="card-gamma")
                    yield MetricCard(title="Vega (ν)", subtitle="Vol Sensitivity", id="card-vega")
                    yield Static(Panel("[bold yellow]H200 GPU[/]\n[dim]Utilization: 92%[/]", title="Compute Status", border_style="blue"))
                
                with Grid(id="plots-grid"):
                    yield PlotextPlot(id="loss-plot", classes="plot-container")
                    yield PlotextPlot(id="price-plot", classes="plot-container")
                    yield PlotextPlot(id="greeks-plot", classes="plot-container")
                    yield PlotextPlot(id="tput-plot", classes="plot-container")
                
                with Vertical(id="log-row"):
                    yield Label("[bold yellow]🧠 LiuClaw Intelligence Stream[/]")
                    yield RichLog(id="agent-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.history = {
            "loss": [], "price": [], "tput": [],
            "delta": [], "gamma": [], "vega": []
        }
        self.step_count = 0
        self.current_plan: Optional[ScenarioPlan] = None
        self.current_step_idx = 0
        
        self.init_plots()
        self.query_one("#agent-log").write("[bold green]System Initialized. H200 GPU Online.[/]")
        self.action_new_scenario()
        self.set_interval(0.5, self.update_data)

    def init_plots(self):
        theme = "dark"
        for pid in ["#loss-plot", "#price-plot", "#greeks-plot", "#tput-plot"]:
            plt = self.query_one(pid).plt
            plt.theme(theme)
            # High-res braille markers
            # Note: marker="braille" might not be supported by all terminals, 
            # but it is the "pixel" equivalent in TUI.
            
        self.query_one("#loss-plot").plt.title("Heston PDE Residual")
        self.query_one("#price-plot").plt.title("Option Price Trend")
        self.query_one("#greeks-plot").plt.title("Greeks (Delta/Vega)")
        self.query_one("#tput-plot").plt.title("Pricing Throughput")

    def action_toggle_sidebar(self):
        sidebar = self.query_one("#sidebar")
        sidebar.toggle_class("hidden")

    def action_toggle_dual(self):
        self.dual_mode = not self.dual_mode
        self.query_one("#brain-panel").dual_mode = self.dual_mode
        status = "ENABLED" if self.dual_mode else "DISABLED"
        self.query_one("#agent-log").write(f"[bold magenta]Brain Overwrite: Dual-Process Engine {status}.[/]")

    def action_new_scenario(self):
        names = ["FOMC Hawk Pivot", "Geopolitical Supply Shock", "Earnings Momentum Run", "Flash Crash"]
        regimes = ["STABLE", "EXPANSION", "CRASH", "JUMP_DIFFUSION"]
        
        new_steps = []
        for _ in range(5):
            new_steps.append(ScenarioStep(
                regime=random.choice(regimes),
                loss_trend=random.choice(["UP", "DOWN", "FLAT"]),
                price_trend=random.choice(["UP", "DOWN", "FLAT"]),
                agent_reasoning=f"LLM: {random.choice(['Skew steepening', 'Gamma exposure rising', 'Volatility clustering detected'])}."
            ))
        
        self.current_plan = ScenarioPlan(name=random.choice(names), steps=new_steps)
        self.current_step_idx = 0
        self.query_one("#agent-log").write(f"[bold cyan]vLLM Scenario Active:[/] {self.current_plan.name}")

    def update_data(self) -> None:
        self.step_count += 1
        
        if self.current_plan:
            step = self.current_plan.steps[self.current_step_idx]
            self.regime = step.regime
            
            # Simulated Math Logic
            if step.loss_trend == "UP": self.pde_loss *= (1.05 + random.random()*0.05)
            elif step.loss_trend == "DOWN": self.pde_loss = max(self.pde_loss * (0.95 if self.dual_mode else 0.98), 0.0001)
            
            if step.price_trend == "UP": self.option_price += random.random() * 0.5
            elif step.price_trend == "DOWN": self.option_price = max(self.option_price - random.random() * 0.5, 1.0)
            
            # Simulated Greeks
            self.delta = 0.5 + 0.2 * (random.random() - 0.5) if self.regime == "STABLE" else 0.5 + 0.4 * (random.random() - 0.5)
            self.vega = 0.1 + 0.05 * random.random() if self.regime == "STABLE" else 0.2 + 0.1 * random.random()
            self.gamma = 0.02 + 0.01 * random.random()

            if self.step_count % 10 == 0:
                self.current_step_idx = (self.current_step_idx + 1) % len(self.current_plan.steps)
                self.query_one("#agent-log").write(f"[dim italic]Regime Update:[/] {step.agent_reasoning}")
        
        self.throughput = 26300 + random.randint(-200, 200)
        self.pde_loss = max(min(self.pde_loss, 50.0), 0.0001)
        
        # Update Histories
        self.history["loss"].append(self.pde_loss)
        self.history["price"].append(self.option_price)
        self.history["tput"].append(self.throughput)
        self.history["delta"].append(self.delta)
        self.history["vega"].append(self.vega)
        
        for k in self.history:
            if len(self.history[k]) > 100: self.history[k].pop(0)
            
        # Update UI Cards
        self.query_one("#card-tput").value = f"{self.throughput:,}"
        self.query_one("#card-loss").value = f"{self.pde_loss:.6f}"
        self.query_one("#card-price").value = f"{self.option_price:.2f}"
        self.query_one("#card-regime").value = self.regime
        self.query_one("#card-delta").value = f"{self.delta:.4f}"
        self.query_one("#card-gamma").value = f"{self.gamma:.4f}"
        self.query_one("#card-vega").value = f"{self.vega:.4f}"
        
        self.refresh_plots()
        self.simulate_brain()

    def simulate_brain(self):
        if not self.dual_mode: return
        brain = self.query_one("#brain-panel")
        brain.s1_active = (self.step_count % 8 == 0)
        brain.s2_active = (self.step_count % 25 == 0)
        if brain.s1_active: self.query_one("#agent-log").write(f"[dim]System 1:[/] Reflexive pruning completed.")
        if brain.s2_active: self.query_one("#agent-log").write(f"[bold cyan]System 2:[/] Strategic review: {self.regime} optimized.")

    def refresh_plots(self):
        # Loss Plot
        lp = self.query_one("#loss-plot")
        lp.plt.clear_data()
        lp.plt.plot(self.history["loss"], color="red")
        lp.refresh()
        
        # Price Plot
        pp = self.query_one("#price-plot")
        pp.plt.clear_data()
        pp.plt.plot(self.history["price"], color="yellow")
        pp.refresh()
        
        # Greeks Plot (Multi-line)
        gp = self.query_one("#greeks-plot")
        gp.plt.clear_data()
        gp.plt.plot(self.history["delta"], color="cyan", label="Delta")
        gp.plt.plot(self.history["vega"], color="magenta", label="Vega")
        gp.refresh()
        
        # Throughput
        tp = self.query_one("#tput-plot")
        tp.plt.clear_data()
        tp.plt.plot(self.history["tput"], color="green")
        tp.refresh()

if __name__ == "__main__":
    app = OpKANDashboard()
    app.run()
