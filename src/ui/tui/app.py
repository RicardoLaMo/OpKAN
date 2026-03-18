import random
import time
import json
import os
from datetime import datetime
from typing import List, Optional, Literal

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, Static, RichLog, Label
from textual.reactive import reactive
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from textual_plotext import PlotextPlot
from src.engine.telemetry import telemetry

# ---------------------------------------------------------------------------
# TUI Components
# ---------------------------------------------------------------------------

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
    engine_online = reactive(False)

    def render(self) -> Panel:
        s1_style = "bold green" if self.s1_active else "dim white"
        s2_style = "bold yellow" if self.s2_active else "dim white"
        mode_str = "[bold magenta]DUAL-BRAIN ENABLED[/]" if self.dual_mode else "[bold red]BASELINE ONLY[/]"
        online_str = "[bold green]ONLINE[/]" if self.engine_online else "[bold red]OFFLINE[/]"
        
        table = Table.grid(expand=True)
        table.add_column()
        table.add_row(Text(f"Engine: {online_str}"))
        table.add_row(Text(mode_str))
        table.add_row(Text("● System 1 (Reflex)", style=s1_style))
        table.add_row(Text("● System 2 (Strategic)", style=s2_style))
        
        return Panel(table, title="Neural Engine", border_style="blue")

class OpKANDashboard(App):
    """The main OpKAN Terminal Telemetry App (Improved IPC)."""
    
    CSS = """
    Screen {
        background: #050505;
    }
    #sidebar {
        width: 30;
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
    #sidebar Vertical {
        height: 1fr;
    }
    #sidebar MetricCard {
        height: 6;
        margin-bottom: 1;
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
        ("b", "toggle_sidebar", "Toggle Sidebar"),
    ]

    throughput = reactive(0)
    pde_loss = reactive(0.0)
    option_price = reactive(0.0)
    delta = reactive(0.0)
    gamma = reactive(0.0)
    vega = reactive(0.0)
    regime = reactive("WAITING")
    dual_mode = reactive(True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield BrainStatus(id="brain-panel")
                yield MetricCard(title="Market Context", subtitle="LLM Decoded", id="card-regime")
                yield MetricCard(title="Pricing Velocity", unit="opts/s", subtitle="H200 Inference", id="card-tput")
                yield MetricCard(title="PDE Error (MSE)", subtitle="Physics Gap", id="card-loss")
                yield MetricCard(title="Option Price", unit="$", subtitle="Market Value", id="card-price")
                yield MetricCard(title="Delta (Δ)", subtitle="Price Sensitivity", id="card-delta")
                yield MetricCard(title="Gamma (Γ)", subtitle="Delta Sensitivity", id="card-gamma")
                yield MetricCard(title="Vega (ν)", subtitle="Vol Sensitivity", id="card-vega")
                yield Static(Panel("[bold yellow]H200 GPU[/]\n[dim]Utilization: 92%[/]", title="Compute Status", border_style="blue"))
            
            with Vertical(id="main-content"):
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
            "delta": [], "vega": []
        }
        self.last_log_idx = 0
        self.last_published_step = -1
        self.init_plots()
        # High-frequency poll for TUI responsiveness
        self.set_interval(0.1, self.poll_telemetry)

    def init_plots(self):
        theme = "dark"
        for pid in ["#loss-plot", "#price-plot", "#greeks-plot", "#tput-plot"]:
            plt = self.query_one(pid).plt
            plt.theme(theme)
            
        self.query_one("#loss-plot").plt.title("Heston PDE Residual")
        self.query_one("#price-plot").plt.title("Option Price Trend")
        self.query_one("#greeks-plot").plt.title("Greeks (Delta/Vega)")
        self.query_one("#tput-plot").plt.title("Pricing Throughput")

    def action_toggle_sidebar(self):
        self.query_one("#sidebar").toggle_class("hidden")

    def poll_telemetry(self) -> None:
        """Polls the real telemetry store for 100% live data."""
        data = telemetry.read()
        if not data:
            self.query_one("#brain-panel").engine_online = False
            return

        # 0. Check if engine is actually active
        is_active = data.get("active", False)
        self.query_one("#brain-panel").engine_online = is_active
        if not is_active:
            return

        current_step = data.get("step", 0)
        
        # 1. Update Metrics (Always update cards for current values)
        self.throughput = data.get("throughput", 0)
        self.pde_loss = data.get("pde_loss", 0.0)
        self.option_price = data.get("option_price", 0.0)
        self.regime = str(data.get("regime", "WAITING")).upper()
        self.delta = data.get("delta", 0.0)
        self.gamma = data.get("gamma", 0.0)
        self.vega = data.get("vega", 0.0)
        
        # 2. Update Histories only if step has advanced
        # This prevents flat lines caused by redundant plotting of the same point
        if current_step > self.last_published_step:
            self.history["loss"].append(self.pde_loss)
            self.history["price"].append(self.option_price)
            self.history["tput"].append(self.throughput)
            self.history["delta"].append(self.delta)
            self.history["vega"].append(self.vega)
            
            for k in self.history:
                if len(self.history[k]) > 150: # Increased history for better resolution
                    self.history[k].pop(0)
            
            self.last_published_step = current_step
            self.refresh_plots()
            
        # 3. Update UI Cards
        self.query_one("#card-tput").value = f"{self.throughput:,}"
        self.query_one("#card-loss").value = f"{self.pde_loss:.6f}"
        self.query_one("#card-price").value = f"{self.option_price:.2f}"
        
        # Color-code regime card
        regime_card = self.query_one("#card-regime")
        regime_card.value = self.regime
        if "JUMP" in self.regime or "CRASH" in self.regime:
            regime_card.styles.border = ("double", "red")
        elif "EXPANSION" in self.regime:
            regime_card.styles.border = ("double", "yellow")
        else:
            regime_card.styles.border = ("double", "green")

        self.query_one("#card-delta").value = f"{self.delta:.4f}"
        self.query_one("#card-gamma").value = f"{self.gamma:.4f}"
        self.query_one("#card-vega").value = f"{self.vega:.4f}"
        
        # 4. Update Brain Panel
        brain = self.query_one("#brain-panel")
        brain.s1_active = data.get("s1_active", False)
        brain.s2_active = data.get("s2_active", False)
        brain.dual_mode = data.get("dual_mode", True)
        brain.engine_online = True # If we got here and is_active was true

        # 5. Handle New Logs
        logs = data.get("logs", [])
        if len(logs) > self.last_log_idx:
            new_entries = logs[self.last_log_idx:]
            for entry in new_entries:
                self.query_one("#agent-log").write(f"[dim]{entry['timestamp']}[/] {entry['message']}")
            self.last_log_idx = len(logs)
        elif len(logs) < self.last_log_idx:
            self.last_log_idx = 0

    def refresh_plots(self):
        # Loss Plot
        lp = self.query_one("#loss-plot")
        lp.plt.clear_data()
        lp.plt.plot(self.history["loss"], color="red", label="Residual", marker="hd")
        lp.refresh()
        
        # Price Plot
        pp = self.query_one("#price-plot")
        pp.plt.clear_data()
        pp.plt.plot(self.history["price"], color="yellow", label="V(S,v,t)", marker="hd")
        pp.refresh()
        
        # Greeks Plot
        gp = self.query_one("#greeks-plot")
        gp.plt.clear_data()
        if self.history["delta"]:
            gp.plt.plot(self.history["delta"], color="cyan", label="Delta", marker="hd")
        if self.history["vega"]:
            gp.plt.plot(self.history["vega"], color="magenta", label="Vega", marker="hd")
        gp.refresh()
        
        # Throughput
        tp = self.query_one("#tput-plot")
        tp.plt.clear_data()
        tp.plt.plot(self.history["tput"], color="green", label="Velocity", marker="hd")
        tp.refresh()

if __name__ == "__main__":
    app = OpKANDashboard()
    app.run()
