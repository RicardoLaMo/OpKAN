import random
import time
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, RichLog, Label
from textual.reactive import reactive
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from textual_plotext import PlotextPlot

class StoryPanel(Static):
    """The static/narrative sidebar describing the OpKAN project."""
    def on_mount(self) -> None:
        self.update_story()

    def update_story(self):
        story_md = """
# 🛡️ OpKAN Project
**Physics-Informed KAN for Options**

## 1. The Math
Solving the **Heston PDE**:
$$\\frac{\\partial V}{\\partial t} + \\dots - rV = 0$$
Using **C2-continuous** B-splines.

## 2. The Architecture
- **PI-KAN**: Core PDE solver.
- **HMM**: Regime detector.
- **LiuClaw**: reasoning Agent.

## 3. Dual-Process Brain
- **System 1**: Reflexive pruning.
- **System 2**: Strategic mutations.
"""
        self.update(Markdown(story_md))

class MetricCard(Static):
    """A small card showing a single metric."""
    value = reactive("0")
    
    def __init__(self, title: str, unit: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.unit = unit

    def render(self) -> Panel:
        return Panel(
            f"[bold cyan]{self.value}[/] {self.unit}",
            title=self.title,
            border_style="green"
        )

class OpKANDashboard(App):
    """The main OpKAN Terminal Telemetry App."""
    
    CSS = """
    Screen {
        background: #121212;
    }
    #sidebar {
        width: 35;
        background: #1a1a1a;
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
        height: 10;
        border-top: dashed #333;
    }
    .plot-container {
        width: 1fr;
        height: 1fr;
        border: solid #333;
        margin: 1;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh Data"),
    ]

    throughput = reactive(0)
    pde_loss = reactive(1.0)
    current_regime = reactive("STABLE")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield StoryPanel()
            with Vertical(id="main-content"):
                with Horizontal(id="metrics-row"):
                    yield MetricCard(title="Throughput", unit="samp/s", id="card-tput")
                    yield MetricCard(title="PDE Loss", id="card-loss")
                    yield MetricCard(title="Regime", id="card-regime")
                
                with Horizontal(id="plots-row"):
                    yield PlotextPlot(title="Loss Convergence", id="loss-plot")
                    yield PlotextPlot(title="Effective Throughput", id="tput-plot")
                
                with Vertical(id="log-row"):
                    yield Label("[bold yellow]🧠 LiuClaw reasoning Stream[/]")
                    yield RichLog(id="agent-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.loss_history = []
        self.tput_history = []
        self.set_interval(1.0, self.update_data)
        
        # Initialize plots
        self.init_plots()
        
        self.query_one("#agent-log").write("[bold green]System Initialized. NVIDIA H200 Ready.[/]")
        self.query_one("#agent-log").write("LiuClaw Agent connecting to vLLM endpoint...")

    def init_plots(self):
        # Loss Plot
        loss_plt = self.query_one("#loss-plot").plt
        loss_plt.theme("dark")
        loss_plt.ylabel("MSE Loss")
        
        # Tput Plot
        tput_plt = self.query_one("#tput-plot").plt
        tput_plt.theme("dark")
        tput_plt.ylabel("Samples/sec")

    def update_data(self) -> None:
        """Simulates data arrival for the storyboard."""
        # 1. Update Reactives
        self.throughput = 26000 + random.randint(-500, 500)
        new_loss = self.pde_loss * 0.95 + random.random() * 0.01
        self.pde_loss = max(new_loss, 0.001)
        
        # 2. Update History
        self.loss_history.append(self.pde_loss)
        self.tput_history.append(self.throughput)
        if len(self.loss_history) > 50:
            self.loss_history.pop(0)
            self.tput_history.pop(0)
            
        # 3. Update Cards
        self.query_one("#card-tput").value = f"{self.throughput:,}"
        self.query_one("#card-loss").value = f"{self.pde_loss:.6f}"
        self.query_one("#card-regime").value = "EXPANSION" if self.pde_loss > 0.1 else "STABLE"
        
        # 4. Refresh Plots
        self.refresh_plots()
        
        # 5. Log random agent mutation
        if random.random() > 0.9:
            ts = datetime.now().strftime("%H:%M:%S")
            self.query_one("#agent-log").write(
                f"[dim]{ts}[/] [bold cyan]LiuClaw:[/] Detected structural shift. [bold yellow]PRUNE[/] Edge L0_N15_to_L1_N4."
            )

    def refresh_plots(self):
        # Update Loss Plot
        loss_plot = self.query_one("#loss-plot")
        loss_plot.plt.clear_data()
        loss_plot.plt.plot(self.loss_history, color="red", label="PDE Residual")
        loss_plot.refresh()
        
        # Update Tput Plot
        tput_plot = self.query_one("#tput-plot")
        tput_plot.plt.clear_data()
        tput_plot.plt.plot(self.tput_history, color="green", label="Throughput")
        tput_plot.refresh()

if __name__ == "__main__":
    app = OpKANDashboard()
    app.run()
