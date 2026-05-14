import pytest
from smart_agent import create_agent  # Your agent factory

@pytest.mark.asyncio
async def test_greeting_response():
    """Test that agent responds appropriately to greeting."""
    agent = create_agent()

    # Simulate user input
    response = await agent.process_text("Hello, I need help with my order")

    # Assert response contains expected content
    assert response is not None
    assert "help" in response.lower() or "order" in response.lower()