import pytest
import os
import json
from src.ui.tui.app import OpKANDashboard, MetricCard
from src.engine.telemetry import telemetry
from textual_plotext import PlotextPlot

@pytest.mark.asyncio
async def test_app_compose_flow():
    """
    Headless test of the App's compose and mount cycle.
    Ensures that the real IPC pipeline works.
    """
    # 1. Mock telemetry data
    test_data = {
        "step": 10,
        "pde_loss": 0.05,
        "option_price": 12.5,
        "delta": 0.6,
        "gamma": 0.03,
        "vega": 0.15,
        "throughput": 26000,
        "regime": "STABLE",
        "logs": [{"timestamp": "12:00:00", "message": "Test log"}],
        "s1_active": True,
        "s2_active": False,
        "dual_mode": True,
        "active": True
    }
    telemetry.write(test_data)

    app = OpKANDashboard()
    async with app.run_test() as pilot:
        assert app.is_running
        
        # Verify presence of key widgets
        assert app.query_one("#loss-plot", PlotextPlot)
        assert app.query_one("#price-plot", PlotextPlot)
        assert app.query_one("#card-tput", MetricCard)
        
        # Trigger polling manually for the test
        app.poll_telemetry()
        
        # Check values after update
        regime = app.query_one("#card-regime", MetricCard).value
        assert regime == "STABLE"
        assert app.query_one("#card-tput", MetricCard).value == "26,000"
        assert app.query_one("#card-loss", MetricCard).value == "0.050000"
        
        await pilot.exit(None)
