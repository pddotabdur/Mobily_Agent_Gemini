"""
Shared fixtures and helpers for the Tawafuq agent test suite.

Each agent test file imports from here via pytest's conftest discovery.
To test a new agent file, add agent-specific fixtures in its own test file
and use make_call_data() / judge_llm from this conftest.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from livekit.plugins import openai

# Make the project root importable so tests can do `from smart_agent import ...`
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env so OPENAI_API_KEY is available before any fixture creates an
# openai.LLM (conftest is imported before test modules).
# override=True so stale shell env vars don't shadow the values in .env.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Suppress verbose LiveKit CLI output (recommended in LiveKit docs)
logging.basicConfig(level=logging.WARNING)
for _noisy in ("livekit", "asyncio", "httpx", "httpcore", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Model configuration — override via env vars for CI / cost control
# ---------------------------------------------------------------------------
# Tests use the direct OpenAI plugin (livekit.plugins.openai), authenticated
# with OPENAI_API_KEY. This bypasses LiveKit Inference quota entirely.
# Model IDs use OpenAI's native format (no `openai/` prefix).
JUDGE_MODEL = os.getenv("TEST_JUDGE_MODEL", "gpt-5.3-chat-latest")
AGENT_MODEL = os.getenv("TEST_AGENT_MODEL", "gpt-4.1")


# ---------------------------------------------------------------------------
# CallData factory
# ---------------------------------------------------------------------------

def make_call_data(**overrides):
    """
    Return a CallData instance wired for testing:
      - sip_ready is pre-set so on_enter() never blocks waiting for SIP dial.
      - Defaults represent a realistic male customer with a 1500 SAR debt.

    Pass keyword overrides to test different customer profiles, amounts, etc.
    """
    from smart_agent import CallData  # imported here to avoid circular imports

    defaults: dict = dict(
        name="محمد الأحمد",
        first_name="محمد",
        gender="male",
        amount="1500",
        national_id_last4="1234",
        phone_number="0501234567",
        services=[
            {
                "type": "إنترنت منزلي",
                "subscription_date": "2024-01-15",
                "service_number_last4": "7890",
            }
        ],
    )
    defaults.update(overrides)
    data = CallData(**defaults)
    data.sip_ready.set()  # Unblock on_enter() immediately — no real SIP needed
    return data


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def std_call_data():
    """Standard male customer with 1500 SAR debt — use in most tests."""
    return make_call_data()


@pytest.fixture
def female_call_data():
    """Female customer variant — tests gender-aware address forms."""
    return make_call_data(name="فاطمة المالكي", first_name="فاطمة", gender="female")


@pytest.fixture
def large_debt_call_data():
    """High-value debt — tests Arabic number pronunciation regression."""
    return make_call_data(amount="15750")


@pytest.fixture
async def judge_llm():
    """
    LLM instance for LLM-based judgment in tests.
    Passed to result.expect.next_event().is_message(...).judge(llm, intent=...).

    Uses the direct OpenAI plugin (OPENAI_API_KEY) instead of LiveKit Inference
    so tests don't consume the LiveKit Inference quota.
    """
    async with openai.LLM(model=JUDGE_MODEL) as llm:
        yield llm
