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
    """A simple container for metrics, updated via .update() in poll_telemetry."""
    def __init__(self, title: str, unit: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.unit = unit

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
    """The main OpKAN Terminal Telemetry App (Bloomberg Trader Edition)."""
    
    # Bloomberg-inspired High-Contrast Palette
    # Background: #000000 (Pure Black)
    # Primary Text: #f7a500 (Amber/Orange)
    # Highlights: #00ffff (Cyan)
    # Positive: #00ff00 (Green)
    # Negative/Alert: #ff0000 (Red)
    # Borders: #333333 (Dark Gray)
    
    CSS = """
    Screen {
        background: #000000;
        color: #f7a500; /* Bloomberg Amber */
    }
    Header {
        background: #111111;
        color: #00ffff;
        border-bottom: solid #333333;
    }
    Footer {
        background: #111111;
        color: #00ffff;
    }
    #sidebar {
        width: 32;
        background: #050505;
        padding: 0 1;
        border-right: double #333333;
    }
    #main-content {
        padding: 0 1;
    }
    MetricCard {
        height: 5;
        margin-bottom: 0;
        border: none;
    }
    #brain-panel {
        height: 7;
        margin-top: 1;
        border: solid #00ffff;
    }
    #plots-grid {
        height: 1fr;
        grid-size: 2 2;
        grid-gutter: 1;
    }
    #log-row {
        height: 12;
        border-top: double #333333;
        background: #050505;
    }
    .plot-container {
        border: solid #1a1a1a;
        background: #020202;
    }
    RichLog {
        background: #000000;
        color: #cccccc;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("b", "toggle_sidebar", "Toggle Sidebar"),
        ("r", "reset_plots", "Reset Plots"),
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
                
                # Market Data Section
                yield Label("\n[bold cyan]── MARKET REGIME ──[/]")
                yield MetricCard(title="STATUS", id="card-regime")
                
                yield Label("\n[bold cyan]── PERFORMANCE ──[/]")
                yield MetricCard(title="VELOCITY", unit="opts/s", id="card-tput")
                yield MetricCard(title="PDE RESID", id="card-loss")
                
                yield Label("\n[bold cyan]── DERIVATIVES ──[/]")
                yield MetricCard(title="PRICE", unit="USD", id="card-price")
                yield MetricCard(title="DELTA (Δ)", id="card-delta")
                yield MetricCard(title="GAMMA (Γ)", id="card-gamma")
                yield MetricCard(title="VEGA (ν)", id="card-vega")
                
                yield Static(Panel("[bold white]H200 CORE[/]\n[green]LOAD: 94%[/] | [cyan]TEMP: 68C[/]", 
                                   title="Hardware", border_style="blue"))
            
            with Vertical(id="main-content"):
                with Grid(id="plots-grid"):
                    yield PlotextPlot(id="loss-plot", classes="plot-container")
                    yield PlotextPlot(id="price-plot", classes="plot-container")
                    yield PlotextPlot(id="greeks-plot", classes="plot-container")
                    yield PlotextPlot(id="tput-plot", classes="plot-container")
                
                with Vertical(id="log-row"):
                    yield Label("[bold cyan]▶ LIVE INTELLIGENCE FEED[/] [dim](LiuClaw Strategic Stream)[/]")
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
        self.set_interval(0.1, self.poll_telemetry)

    def init_plots(self):
        # Bloomberg plots often have very simple black backgrounds
        for pid in ["#loss-plot", "#price-plot", "#greeks-plot", "#tput-plot"]:
            plt = self.query_one(pid).plt
            plt.theme("dark")
            plt.canvas_color("black")
            plt.axes_color("black")
            plt.ticks_color("#555555")
            
        self.query_one("#loss-plot").plt.title("Heston Residual (L2)")
        self.query_one("#price-plot").plt.title("Surface Valuation (V)")
        self.query_one("#greeks-plot").plt.title("Sensitivities (Δ/ν)")
        self.query_one("#tput-plot").plt.title("Compute Throughput")

    def action_toggle_sidebar(self):
        self.query_one("#sidebar").toggle_class("hidden")

    def action_reset_plots(self):
        for k in self.history: self.history[k] = []
        self.refresh_plots()

    def _draw_waiting_state(self):
        """Draws placeholders when engine is offline."""
        for cid in ["#card-regime", "#card-tput", "#card-loss", "#card-price", "#card-delta", "#card-gamma", "#card-vega"]:
            card = self.query_one(cid)
            card.update(Panel("[dim]WAITING...[/]", title=card.title, border_style="#333333"))

    def poll_telemetry(self) -> None:
        data = telemetry.read()
        if not data:
            self.query_one("#brain-panel").engine_online = False
            self._draw_waiting_state()
            return

        is_active = data.get("active", False)
        self.query_one("#brain-panel").engine_online = is_active
        if not is_active:
            self._draw_waiting_state()
            return

        current_step = data.get("step", 0)
        self.throughput = data.get("throughput", 0)
        self.pde_loss = data.get("pde_loss", 0.0)
        self.option_price = data.get("option_price", 0.0)
        self.regime = str(data.get("regime", "WAITING")).upper()
        self.delta = data.get("delta", 0.0)
        self.gamma = data.get("gamma", 0.0)
        self.vega = data.get("vega", 0.0)
        
        if current_step > self.last_published_step:
            self.history["loss"].append(self.pde_loss)
            self.history["price"].append(self.option_price)
            self.history["tput"].append(self.throughput)
            self.history["delta"].append(self.delta)
            self.history["vega"].append(self.vega)
            
            for k in self.history:
                if len(self.history[k]) > 100: self.history[k].pop(0)
            
            self.last_published_step = current_step
            self.refresh_plots()
            
        # ── Bloomberg Metric Styling ──
        # Update cards with specific styles
        self.query_one("#card-tput").update(Panel(f"[bold white]{self.throughput:,}[/] {self.query_one('#card-tput').unit}", title="VELOCITY", border_style="#333333"))
        
        # Loss Card: Red if high, Green if low
        loss_color = "red" if self.pde_loss > 0.1 else "green"
        self.query_one("#card-loss").update(Panel(f"[{loss_color}]{self.pde_loss:.6f}[/]", title="PDE RESID", border_style="#333333"))
        
        self.query_one("#card-price").update(Panel(f"[bold cyan]${self.option_price:.2f}[/]", title="PRICE", border_style="#333333"))
        
        # Regime Card: Professional Bloomberg Color Logic
        regime_card = self.query_one("#card-regime")
        if "JUMP" in self.regime or "CRASH" in self.regime:
            reg_style = "bold white on red"
            reg_border = "red"
        elif "EXPANSION" in self.regime:
            reg_style = "bold black on #f7a500" # Amber background
            reg_border = "#f7a500"
        else:
            reg_style = "bold green"
            reg_border = "green"
        
        regime_card.update(Panel(f"[{reg_style}] {self.regime} [/]", title="STATUS", border_style=reg_border))

        self.query_one("#card-delta").update(Panel(f"[cyan]{self.delta:.4f}[/]", title="DELTA (Δ)", border_style="#333333"))
        self.query_one("#card-gamma").update(Panel(f"[white]{self.gamma:.4f}[/]", title="GAMMA (Γ)", border_style="#333333"))
        self.query_one("#card-vega").update(Panel(f"[magenta]{self.vega:.4f}[/]", title="VEGA (ν)", border_style="#333333"))
        
        # ── Brain Panel ──
        brain = self.query_one("#brain-panel")
        brain.s1_active = data.get("s1_active", False)
        brain.s2_active = data.get("s2_active", False)
        brain.dual_mode = data.get("dual_mode", True)
        brain.engine_online = True

        # ── Logs ──
        logs = data.get("logs", [])
        if len(logs) > self.last_log_idx:
            for entry in logs[self.last_log_idx:]:
                # Bloomberg log style: Cyan timestamp, White text
                self.query_one("#agent-log").write(f"[cyan]{entry['timestamp']}[/] [white]{entry['message']}[/]")
            self.last_log_idx = len(logs)

    def refresh_plots(self):
        # Loss Plot
        lp = self.query_one("#loss-plot")
        lp.plt.clear_data()
        lp.plt.plot(self.history["loss"], color="red", label="Residual")
        lp.refresh()
        
        # Price Plot
        pp = self.query_one("#price-plot")
        pp.plt.clear_data()
        pp.plt.plot(self.history["price"], color="yellow", label="V(S,v,t)")
        pp.refresh()
        
        # Greeks Plot
        gp = self.query_one("#greeks-plot")
        gp.plt.clear_data()
        if self.history["delta"]:
            gp.plt.plot(self.history["delta"], color="cyan", label="Delta")
        if self.history["vega"]:
            gp.plt.plot(self.history["vega"], color="magenta", label="Vega")
        gp.refresh()
        
        # Throughput
        tp = self.query_one("#tput-plot")
        tp.plt.clear_data()
        tp.plt.plot(self.history["tput"], color="green", label="Velocity")
        tp.refresh()

if __name__ == "__main__":
    app = OpKANDashboard()
    app.run()
