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

#  tests/conftest.py — shared fixtures for the whole suite
#   - judge_llm fixture — inference.LLM for .judge() calls
#   - make_call_data(**overrides) — creates CallData with sip_ready pre-set (critical: without this,
#   Stage1RightPartyAgent.on_enter() blocks forever waiting for SIP)
#   - std_call_data, female_call_data, large_debt_call_data fixtures for common profiles
#   - TEST_JUDGE_MODEL / TEST_AGENT_MODEL env vars for model swaps

#   tests/test_smart_agent.py — 39 tests across 5 classes

#   ┌──────────────────────────┬──────────────────────────────────────────────────────────────────────────┐
#   │          Class           │                              What it tests                               │
#   ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
#   │ TestStage1RightParty     │ Identity confirmation, wrong party, busy, DNC, Arabic greeting format,   │
#   │                          │ gender-aware address                                                     │
#   ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
#   │ TestStage1IDVerification │ Yes/no digit confirmation, id_denied outcome, refusal handling           │
#   ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
#   │ TestStage2DebtIntro      │ Reason capture, already-paid, dispute, amount in Arabic words regression │
#   │                          │  (1500 and 15750 SAR)                                                    │
#   ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
#   │ TestStage3Negotiation    │ Full/two-step commit, flat refusal, consequences line content, no        │
#   │                          │ threatening language                                                     │
#   ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
#   │ TestStage4Recap          │ Confirmed, renegotiation, SADAD biller code 005, callback phrasing       │
#   ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
#   │ TestFullConversations    │ 6 multi-turn paths with JudgeGroup (accuracy, relevancy,                 │
#   │                          │ task_completion, tool_use, safety, coherence)                            │
#   ├──────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
#   │ TestMockedTools          │ Error paths via mock_tools, partial mobile number rejection              │
#   └──────────────────────────┴──────────────────────────────────────────────────────────────────────────┘

#   To run:
#   uv run pytest tests/test_smart_agent.py -v           # all tests
#   uv run pytest tests/ -v -k "stage1"                 # one class
#   uv run pytest tests/ -v -k "full_conversations"     # JudgeGroup regression only

#   To add tests for a new agent file — just create tests/test_<agent>.py, import make_call_data from
#   conftest, and follow the same AgentSession + session.run() pattern.
