import pytest
from textual.widgets import Header, Footer
from src.ui.tui.app import OpKANDashboard, MetricCard, StoryPanel
from textual_plotext import PlotextPlot

def test_tui_component_instantiation():
    """Verify that all custom TUI components can be instantiated without error."""
    story = StoryPanel()
    card = MetricCard(title="Test", unit="unit")
    app = OpKANDashboard()
    
    assert app.title == "OpKANDashboard"
    assert card.title == "Test"

@pytest.mark.asyncio
async def test_app_compose_flow():
    """
    Headless test of the App's compose and mount cycle.
    Ensures that the TypeError reported by the user is resolved.
    """
    app = OpKANDashboard()
    async with app.run_test() as pilot:
        # If it reaches here without raising TypeError, the compose logic is valid
        assert app.is_running
        
        # Verify presence of key widgets
        assert app.query_one("#loss-plot", PlotextPlot)
        assert app.query_one("#tput-plot", PlotextPlot)
        assert app.query_one("#card-tput", MetricCard)
        assert app.query_one("#agent-log")
        
        # Trigger one data update manually to populate values
        app.update_data()
        
        # Check values after update
        regime = app.query_one("#card-regime", MetricCard).value
        assert regime in ["STABLE", "EXPANSION", "CRASH", "JUMP_DIFFUSION", "INITIALIZING"]
        assert int(app.query_one("#card-tput", MetricCard).value.replace(",", "")) > 0
        
        # Verify log is working
        assert app.query_one("#agent-log")
        
        await pilot.exit(None)
