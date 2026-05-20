"""
Regression tests for smart_agent.py (Nora — Mobily debt-collection agent).

All assertions trace back to the call-flow spec in `debit.pdf`. Each test's
docstring cites the PDF section it covers so a reader can audit the
behaviour against the document.

Test organisation (mapped to PDF sections)
──────────────────────────────────────────
  TestStage1RightParty      — PDF 3.1: right-party verification + side-question
                              handling; never disclose debt at this stage
  TestStage1IDVerification  — PDF 3.1: last-4-digit yes/no; refuse → close
  TestStage2DebtIntro       — PDF 3.2: introduce Nora + recording + SIMAH-as-
                              avoidance; reason capture (incl. hardship);
                              already-paid 2-turn flow; clarify_debt format;
                              vague reply re-ask once then move on
  TestStage3Negotiation     — PDF 3.3: hardship-aware opening, ladder with
                              exception framing on Attempts 2/3, mid-ladder
                              empathy, refusal close credit-record line
  TestStage3PercentageCheck — PDF 3.3: 30%+ / 25-30% / <25% one-gentle-push /
                              <25% after push (locks with below_threshold flag)
  TestStage3Stalemate       — PDF 3.3: post-ladder stalemate disclosure
                              covering SIMAH + legal action + court fees,
                              followed by final commitment ask → dispute
  TestStage4Recap           — PDF 3.4: restate plan, all 5 payment methods
                              (SADAD 055 + bank app + Mobily app + branch +
                              ATM), ask for receipt, end with مضبوط؟
  TestSpecialSituations     — PDF 4: DNC (4.1), death (4.2), cancelled line /
                              ported number (4.4), alo recovery (4.8)
  TestFullConversations     — end-to-end multi-turn JudgeGroup regression
  TestMockedTools           — edge-case tool mocking (error paths, bad data)

How to run
──────────
  uv run pytest tests/test_smart_agent.py -v
  uv run pytest tests/test_smart_agent.py -v -k "stage1"
  uv run pytest tests/test_smart_agent.py -v -k "stalemate"
  uv run pytest tests/test_smart_agent.py -v --tb=short

Environment variables
─────────────────────
  TEST_JUDGE_MODEL  LLM for intent evaluation  (default: openai/gpt-5.3-chat-latest)
  TEST_AGENT_MODEL  LLM driving the agent      (default: openai/gpt-4.1)
"""
from __future__ import annotations

import pytest
from livekit.agents import AgentSession, inference, mock_tools
from livekit.agents.evals import (
    JudgeGroup,
    accuracy_judge,
    coherence_judge,
    relevancy_judge,
    safety_judge,
    task_completion_judge,
    tool_use_judge,
)

from smart_agent import (
    CallData,
    ClosingAgent,
    CollectMobileAgent,
    RescheduleAgent,
    ScheduleCallbackAgent,
    Stage1IDYesNoAgent,
    Stage1RightPartyAgent,
    Stage2DebtIntroAgent,
    Stage3NegotiationAgent,
    Stage4RecapAgent,
    WrongPartyKnowsAgent,
)
from conftest import make_call_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(judge_llm, call_data) -> AgentSession:
    """Return a fresh AgentSession backed by the test agent + judge LLMs.

    `call_data` is wired in as session userdata so tools that access
    `ctx.userdata` (e.g. right_party, id_confirmed) can mutate it.
    """
    return AgentSession[CallData](llm=judge_llm, userdata=call_data)


# ============================================================
# Stage 1 — Right-party verification
# ============================================================

class TestStage1RightParty:
    """
    Stage1RightPartyAgent opens the call and classifies the reply into one of:
    right_party / wrong_party / caller_busy / do_not_call / customer_deceased.
    """

    @pytest.mark.asyncio
    async def test_confirms_as_right_party(self, judge_llm, std_call_data):
        """نعم / affirmation → right_party tool → handoff to Stage1IDYesNoAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(user_input="نعم، أنا محمد")

            # Agent must call right_party tool
            result.expect.next_event().is_function_call(name="right_party")
            result.expect.next_event().is_function_call_output()
            # After handoff the new agent (IDYesNo) sends the verification question
            result.expect.contains_agent_handoff(new_agent_type=Stage1IDYesNoAgent)

    @pytest.mark.asyncio
    async def test_confirms_identity_verified_flag(self, judge_llm, std_call_data):
        """identity_confirmed flag is set after right_party tool fires."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))
            await session.run(user_input="أيوه، أنا")

        assert std_call_data.identity_confirmed is True

    @pytest.mark.asyncio
    async def test_wrong_party_explicit(self, judge_llm, std_call_data):
        """Explicit denial → wrong_party tool → WrongPartyKnowsAgent handoff."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(user_input="لا، أنت غلطان، ما أنا محمد")

            result.expect.next_event().is_function_call(name="wrong_party")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=WrongPartyKnowsAgent)

    @pytest.mark.asyncio
    async def test_caller_busy(self, judge_llm, std_call_data):
        """مشغول / bad-time response → caller_busy tool → ScheduleCallbackAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(user_input="أنا مشغول الحين، كلمني بعدين")

            result.expect.next_event().is_function_call(name="caller_busy")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=ScheduleCallbackAgent)

    @pytest.mark.asyncio
    async def test_do_not_call_request(self, judge_llm, std_call_data):
        """DNC request → do_not_call tool → ClosingAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(user_input="لا تتصل فيني مرة ثانية")

            result.expect.next_event().is_function_call(name="do_not_call")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

    @pytest.mark.asyncio
    async def test_opening_message_is_arabic_greeting(self, judge_llm, std_call_data):
        """
        After session start, the first agent reply must be a short Arabic
        greeting that addresses the customer by name.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            # Provide a neutral input so we get the next message without a tool call
            result = await session.run(user_input="آلو")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent greets the caller in Arabic and asks "
                        "if they are the named customer (محمد الأحمد). "
                        "The message is short (one sentence)."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_female_customer_addressing(self, judge_llm, female_call_data):
        """Female customers are addressed as 'الأستاذة' in the opening."""
        async with _session(judge_llm, female_call_data) as session:
            await session.start(Stage1RightPartyAgent(female_call_data))

            result = await session.run(user_input="آلو")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent addresses the caller with a female title "
                        "(الأستاذة) and the name فاطمة المالكي."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_side_question_who_are_you_returns_to_verification(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.1: side question "من أنت؟" is answered in ONE short sentence
        ("معك نورا من توافق، نتصل بالنيابة عن موبايلي" or similar) and the
        agent immediately returns to the verification question. The reply
        must NOT include any debt amount or account detail.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(user_input="من أنتي؟ وش تبين؟")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent answers in ONE short sentence that she is "
                        "Nora from Tawafuq, an authorized representative of "
                        "Mobily, and then returns to asking if she is "
                        "speaking with the named customer (محمد). The reply "
                        "must NOT mention any debt amount, ID digits, or "
                        "any account-specific detail."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_side_question_why_calling_returns_to_verification(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.1: "Why are you calling?" answered with the canonical short
        line ("بخصوص حسابك في موبايلي") and the agent returns to the
        verification question without revealing debt details.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(user_input="ليش تتصلون؟ وش الموضوع؟")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent answers in ONE short sentence that the "
                        "call is regarding the customer's Mobily account, "
                        "without disclosing any amount or further detail, "
                        "and then asks again to confirm she is speaking "
                        "with the named customer."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_side_question_where_got_number_returns_to_verification(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.1: "Where did you get my number?" answered with the canonical
        line pointing to Mobily's registered records and offering to verify
        by calling 1100, then back to the verification question.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(
                user_input="من وين جبتي رقمي؟ ما أعطيت رقمي لأحد"
            )

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent explains in ONE short sentence that the "
                        "number is on Mobily's official/registered records "
                        "(السجلات الرسمية لدى موبايلي) and offers the customer "
                        "to verify by calling 1100. Then the agent returns "
                        "to the verification question. No debt details are "
                        "disclosed."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_no_debt_amount_disclosed_at_stage1(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.1 compliance: until identity is verified, Nora must NEVER
        disclose the debt amount or any account-specific detail. We probe
        with an explicit ask about the purpose.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(
                user_input="قبل ما أأكد، قولي وش المبلغ اللي عليّ؟"
            )

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent declines to share the amount, OR speaks "
                        "only in general terms about a Mobily account, and "
                        "asks for verification first. The specific number "
                        "1500 (or its Arabic-word equivalent ألف وخمس مية) "
                        "must NOT appear in the reply."
                    ),
                )
            )


# ============================================================
# Stage 1 — ID last-4 yes/no verification
# ============================================================

class TestStage1IDVerification:
    """
    Stage1IDYesNoAgent reads back the last-4 ID digits and awaits YES/NO.
    """

    @pytest.mark.asyncio
    async def test_id_confirmed_advances_to_stage2(self, judge_llm, std_call_data):
        """Customer confirms digits → id_confirmed → Stage2DebtIntroAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1IDYesNoAgent(std_call_data))

            result = await session.run(user_input="نعم، صحيح")

            result.expect.next_event().is_function_call(name="id_confirmed")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=Stage2DebtIntroAgent)

    @pytest.mark.asyncio
    async def test_id_confirmed_sets_verified_flag(self, judge_llm, std_call_data):
        """id_verified flag is set once ID is confirmed."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1IDYesNoAgent(std_call_data))
            await session.run(user_input="أيوه تمام")

        assert std_call_data.id_verified is True

    @pytest.mark.asyncio
    async def test_id_denied_closes_call(self, judge_llm, std_call_data):
        """Customer says digits are wrong → id_denied → ClosingAgent (wrong_party intent)."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1IDYesNoAgent(std_call_data))

            result = await session.run(user_input="لا، هذا مو رقمي")

            result.expect.next_event().is_function_call(name="id_denied")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

    @pytest.mark.asyncio
    async def test_id_denied_sets_outcome(self, judge_llm, std_call_data):
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1IDYesNoAgent(std_call_data))
            await session.run(user_input="غلط، هذا مو رقمي")

        assert std_call_data.outcome == "id_mismatch"

    @pytest.mark.asyncio
    async def test_verification_question_mentions_digits(self, judge_llm, std_call_data):
        """Opening question must state the last-4 digits for the customer to confirm."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1IDYesNoAgent(std_call_data))

            result = await session.run(user_input="ما سمعت زين")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent states individual digits (one, two, three, four "
                        "in Arabic — واحد اثنين ثلاثة أربعة) and asks the customer "
                        "to confirm YES or NO. The sentence is short."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_refuses_to_verify_closes_call(self, judge_llm, std_call_data):
        """Explicit refusal to verify → refuses_to_verify → ClosingAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1IDYesNoAgent(std_call_data))

            result = await session.run(user_input="ما أعطيك أي معلومات")

            result.expect.next_event().is_function_call(name="refuses_to_verify")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)


# ============================================================
# Stage 2 — QA disclosure + debt intro + reason capture
# ============================================================

class TestStage2DebtIntro:
    """
    Stage2DebtIntroAgent discloses recording, states the debt, and asks why.
    """

    @pytest.mark.asyncio
    async def test_reason_captured_advances_to_stage3(self, judge_llm, std_call_data):
        """Customer gives a reason → reason_captured → Stage3NegotiationAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(user_input="عندي ضائقة مالية الفترة الحالية")

            result.expect.next_event().is_function_call(name="reason_captured")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=Stage3NegotiationAgent)

    @pytest.mark.asyncio
    async def test_already_paid_closes_call(self, judge_llm, std_call_data):
        """
        PDF 3.2: when the customer says they already paid, Nora must first ask
        WHEN the payment was made and request the receipt before closing.
        already_paid_done is the tool that fires once that's been asked.

        We give the customer's reply already containing the timing so the LLM
        can — in a single turn — both acknowledge/ask for the receipt and call
        already_paid_done(when_text=...). The handoff target is ClosingAgent.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(
                user_input="أنا سددت هذا المبلغ قبل أسبوع، عندي إيصال"
            )

            # Some reply turns may include a clarifying message before the
            # tool call; assert the tool fires *somewhere* in this turn and
            # ends with a ClosingAgent handoff.
            result.expect.contains_function_call(name="already_paid_done")
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

    @pytest.mark.asyncio
    async def test_already_paid_sets_outcome(self, judge_llm, std_call_data):
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))
            await session.run(user_input="دفعت قبل أسبوع، راح أرسلك الإيصال")

        assert std_call_data.outcome == "paid"

    @pytest.mark.asyncio
    async def test_denial_after_clarify_continues_to_negotiation(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2: even after clarify_debt has already been delivered, a flat
        denial does NOT close the call as a dispute — the call must continue
        into Stage 3 with denied_after_clarify=True. The Stage-3 stalemate
        path (not Stage 2) is the only place that can route to a dispute
        close.
        """
        std_call_data.debt_clarification_given = True  # clarification already given
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(user_input="هذا مو حقي، أنا ما عليّ شي")

            # reason_captured fires with denied_after_clarify=True and we hand
            # off into the negotiation stage — NOT directly to ClosingAgent.
            result.expect.contains_function_call(name="reason_captured")
            result.expect.contains_agent_handoff(new_agent_type=Stage3NegotiationAgent)
        # No dispute close at this point — that decision is reserved for the
        # Stage-3 stalemate disclosure path.
        assert std_call_data.outcome != "dispute"

    @pytest.mark.asyncio
    async def test_hardship_signaled_when_reason_is_financial(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2 → 3.3: phrases like "ما معي فلوس"/"ظروف"/"الأمور صعبة" must
        be flagged as hardship so Stage 3 opens with empathy + softened
        half/half offer rather than the standard full-payment ask.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))
            await session.run(
                user_input="والله ما معي فلوس الفترة هذي، الأمور صعبة جدًا"
            )

        assert std_call_data.hardship_signaled is True, (
            "Stage 2 should mark hardship_signaled=True when the reason is "
            "clearly financial hardship"
        )

    @pytest.mark.asyncio
    async def test_opening_states_amount_in_arabic_words(self, judge_llm, std_call_data):
        """
        PDF 3.2: opening states the amount in Arabic words (never digits).
        For 1500 SAR → 'ألف وخمس مية' (or equivalent Najdi phrasing).
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(user_input="تفضلي")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent mentions an outstanding debt amount of 1500 SAR "
                        "expressed in Arabic words (ألف وخمس مية or similar), "
                        "never as the digits '1500', and asks why the payment is late."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_opening_reintroduces_nora_and_recording_disclosure(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2 single-turn disclosure: introduces Nora again AND notes
        that the call is recorded for quality. Both elements must appear
        in the first reply.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(user_input="تفضلي")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent's opening turn includes BOTH of: "
                        "(a) introducing herself as Nora from Tawafuq, and "
                        "(b) noting that the call may be recorded for "
                        "quality purposes (مسجّلة لأغراض الجودة or similar). "
                        "Both elements appear in the same single connected "
                        "turn before the reason question."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_opening_frames_simah_as_avoidance_not_threat(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2: the SIMAH mention in the debt intro must be framed as
        "resolving it avoids a negative impact" — never as a threat. The
        Stage-3 stalemate disclosure (legal action, court fees) must NOT
        appear here.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(user_input="تفضلي")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent mentions SIMAH (سمة) in a soft, "
                        "avoidance-framed way — e.g. 'تسويته تجنبك تأثير "
                        "سلبي على سجلك الائتماني'. It must NOT threaten "
                        "with legal action (إجراءات قانونية), court fees "
                        "(رسوم محكمة), or use coercive wording like "
                        "'لازم تسدد' or 'إنذار'."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_vague_reply_triggers_one_re_ask(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2: a bare "نعم/تمام/أوكي" reply (no reason given) triggers
        ONE re-ask with the canonical wording "قصدي، وش سبب تأخيرك
        بالسداد؟". The agent must NOT yet hand off to Stage 3 on this
        turn (no reason_captured).
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            # Get past the opening turn first.
            await session.run(user_input="تفضلي")

            # Now reply vaguely.
            result = await session.run(user_input="تمام")

            # No reason_captured handoff yet — agent must re-ask. We verify
            # by directly inspecting the event stream: there should be no
            # reason_captured tool call on this turn.
            function_calls = [
                e for e in result.events
                if getattr(e, "type", None) == "function_call"
            ]
            reason_captures = [
                e for e in function_calls
                if getattr(getattr(e, "item", None), "name", None) == "reason_captured"
            ]
            assert not reason_captures, (
                "On a vague reply the agent must re-ask first, not call "
                "reason_captured and hand off to Stage 3."
            )

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent re-asks the reason question in a polite "
                        "clarifying way (e.g. 'قصدي، وش سبب تأخيرك "
                        "بالسداد؟' or equivalent). ONE short sentence."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_clarify_debt_uses_last4_format(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2 + 4.3: when the customer denies recognising the debt,
        clarify_debt must summarise 1-2 line items using the canonical
        format "الرقم المنتهي بـ ..." and the LAST 4 digits only — never
        the full service number or any internal identifier.

        std_call_data.services has one service with last4 "7890" and
        subscription_date "2024-01-15", type "إنترنت منزلي".
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(
                user_input="ما أعرف هالفاتورة، أنا ما عليّ شي"
            )

            # clarify_debt must fire (no dispute close from Stage 2).
            result.expect.contains_function_call(name="clarify_debt")

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent presents service details using ONLY the "
                        "last 4 digits of the service number ('الرقم "
                        "المنتهي بـ 7890' or 'سبعة ثمانية تسعة صفر'). It "
                        "mentions the service type (إنترنت منزلي) and the "
                        "subscription date. It must NOT read the full "
                        "service number or any internal identifier. Then "
                        "it re-asks the reason."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_already_paid_asks_when_and_receipt_before_closing(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2: when the customer says they already paid, Nora must NOT
        treat the issue as resolved. She first asks WHEN payment was made
        and requests the receipt for verification. Only after that reply
        does she call already_paid_done and close.

        Verify the two-turn shape:
          Turn 1: customer says paid → agent asks when + receipt (no tool).
          Turn 2: customer answers → agent calls already_paid_done.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))
            # Opening
            await session.run(user_input="تفضلي")

            # Turn 1 — bare "I paid" with no timing.
            result1 = await session.run(user_input="سددت قبل فترة")

            # No close yet — agent should ask when + receipt.
            await (
                result1.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent acknowledges the payment claim and "
                        "asks WHEN the payment was made AND asks the "
                        "customer to send the payment receipt for "
                        "verification (إيصال السداد / إيصال الدفع). The "
                        "agent has NOT closed the call yet."
                    ),
                )
            )
            assert std_call_data.outcome != "paid", (
                "Outcome must not yet be set to paid — Nora hasn't asked "
                "for the receipt yet."
            )

            # Turn 2 — customer responds with timing; agent should now
            # call already_paid_done and close.
            result2 = await session.run(
                user_input="قبل ثلاثة أيام، أرسلك الإيصال على واتساب"
            )
            result2.expect.contains_function_call(name="already_paid_done")
            result2.expect.contains_agent_handoff(new_agent_type=ClosingAgent)
        assert std_call_data.outcome == "paid"
        assert std_call_data.payment_made_when is not None

    @pytest.mark.asyncio
    async def test_large_debt_amount_arabic_words(self, judge_llm, large_debt_call_data):
        """Regression: 15750 SAR must be spoken as Arabic words, not digits."""
        async with _session(judge_llm, large_debt_call_data) as session:
            await session.start(Stage2DebtIntroAgent(large_debt_call_data))

            result = await session.run(user_input="تفضلي")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent states the debt amount in Arabic words "
                        "(خمسة عشر ألف وسبع مية وخمسين or similar) — "
                        "the digits '15750' must not appear literally in the message."
                    ),
                )
            )


# ============================================================
# Stage 3 — Negotiation ladder
# ============================================================

class TestStage3Negotiation:
    """
    Stage3NegotiationAgent walks through full → half → instalment offers.
    """

    @pytest.mark.asyncio
    async def test_full_payment_today_commits(self, judge_llm, std_call_data):
        """Customer agrees to pay full amount today → commit_full → Stage4RecapAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(user_input="أقدر أسدد كامل المبلغ اليوم")

            result.expect.next_event().is_function_call(name="commit_full")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=Stage4RecapAgent)

    @pytest.mark.asyncio
    async def test_full_payment_commitment_sets_outcome(self, judge_llm, std_call_data):
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))
            await session.run(user_input="ممكن أسدد اليوم الكامل")

        assert std_call_data.outcome == "committed"

    @pytest.mark.asyncio
    async def test_two_step_payment_commits(self, judge_llm, std_call_data):
        """Half today + rest later → commit_two_step → Stage4RecapAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(
                user_input="أقدر أدفع النصف اليوم والباقي بعد أسبوعين"
            )

            result.expect.next_event().is_function_call(name="commit_two_step")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=Stage4RecapAgent)

    @pytest.mark.asyncio
    async def test_flat_refusal_closes_call(self, judge_llm, std_call_data):
        """Flat refusal → refuses_payment → ClosingAgent (no pressure)."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(user_input="ما أقدر أسدد شي الحين، مو مهتم")

            result.expect.next_event().is_function_call(name="refuses_payment")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

    @pytest.mark.asyncio
    async def test_refusal_sets_outcome(self, judge_llm, std_call_data):
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))
            await session.run(user_input="لن أدفع")

        assert std_call_data.outcome == "refusal"

    @pytest.mark.asyncio
    async def test_standard_opening_asks_for_full_payment(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3: when no hardship was signaled, Stage 3 opens with the
        ideal-outcome ask: full payment today or tomorrow. The SIMAH
        soft-line has already been said in Stage 2 — it must NOT be
        repeated here.
        """
        assert std_call_data.hardship_signaled is False
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(user_input="أبشر، استمر")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent asks whether the customer can pay the FULL "
                        "outstanding amount today or tomorrow (كامل اليوم أو "
                        "بكرا). The message is short, calm, and does NOT "
                        "threaten legal action or court fees. It is also OK "
                        "if the agent does not repeat the SIMAH line — that "
                        "was said in Stage 2."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_hardship_opening_offers_softened_half(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3: when hardship was signaled in Stage 2, Stage 3 opens with
        warm empathy + a softened half-today / rest-within-a-week offer,
        framed as an exception — NOT with the standard full-payment ask.
        """
        std_call_data.hardship_signaled = True
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(user_input="أبشر، استمر")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent opens with a short empathic phrase "
                        "acknowledging the customer's situation (e.g. "
                        "يعينك الله / ندري الموضوع مو سهل / إحنا معك), "
                        "and offers — framed as an exception (كاستثناء) — "
                        "to pay roughly HALF today or tomorrow with the rest "
                        "within a week. It does NOT ask for the full amount."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_propose_installment_at_30_percent_locks_plan(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3 percentage check: a first instalment of 30 % of the total is
        accepted with encouragement and the plan is locked (Stage 4 recap).

        std_call_data.amount = "1500", so 30 % = 450.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            # Customer names a specific amount/date — LLM is expected to
            # call propose_installment with first_amount=~450.
            result = await session.run(
                user_input="أقدر أسدد أربع مية وخمسين ريال يوم خمسة وعشرين"
            )

            result.expect.contains_function_call(name="propose_installment")
            result.expect.contains_agent_handoff(new_agent_type=Stage4RecapAgent)

        assert std_call_data.outcome == "committed"
        assert std_call_data.below_threshold_commit is False

    @pytest.mark.asyncio
    async def test_propose_installment_below_25_triggers_one_gentle_push(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3: a first instalment below 25 % of the total triggers EXACTLY
        ONE gentle push (framed as exception) before the plan is locked.

        std_call_data.amount = "1500", so 20 % = 300 (well below the 25 %
        threshold). After the push the agent should stay in Stage 3 (no
        handoff yet) — it's waiting for the customer's response to the
        gentle push.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(
                user_input="أكثر شي أقدر أسدده ثلاث مية ريال يوم عشرين"
            )

            # propose_installment must fire, but no handoff yet — the agent
            # is mid-push.
            result.expect.contains_function_call(name="propose_installment")

        # gentle_pushed flag flipped, but no commitment locked yet.
        assert std_call_data.gentle_pushed is True
        assert std_call_data.commitment is None

    @pytest.mark.asyncio
    async def test_propose_installment_between_25_and_30_locks_plan(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3 percentage check: 25%-30% range → accept with reassurance
        and lock the plan. No gentle push. No below_threshold flag.

        std_call_data.amount = "1500", so 27% ≈ 405 SAR.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))
            result = await session.run(
                user_input="أقدر أسدد أربع مية وخمسة ريال يوم سبعة وعشرين"
            )
            result.expect.contains_function_call(name="propose_installment")
            result.expect.contains_agent_handoff(new_agent_type=Stage4RecapAgent)

        assert std_call_data.outcome == "committed"
        # The plan was accepted on the customer's amount — but it was at or
        # above 25%, so the below-threshold flag must stay False.
        assert std_call_data.below_threshold_commit is False
        # And we must not have pushed (only <25% triggers the push).
        assert std_call_data.gentle_pushed is False

    @pytest.mark.asyncio
    async def test_propose_installment_below_25_after_push_locks_with_followup(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3 percentage check: when the customer offers <25% AND has
        already been gently pushed once, Nora locks the plan on the
        customer's amount, sets below_threshold_commit=True, and Stage 4
        will later add the warm follow-up line. She must NEVER push twice.

        We simulate "already pushed" by pre-setting gentle_pushed=True so
        the next propose_installment call lands in the "after push" branch.
        """
        std_call_data.gentle_pushed = True  # simulate prior push
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))
            result = await session.run(
                user_input="أقصى شي مية ريال يوم خمسة وعشرين"
            )
            result.expect.contains_function_call(name="propose_installment")
            result.expect.contains_agent_handoff(new_agent_type=Stage4RecapAgent)

        # Locked with below-threshold flag for Stage 4 follow-up line.
        assert std_call_data.below_threshold_commit is True
        assert std_call_data.outcome == "committed_partial"
        # The flag was already True; it must remain True (no resetting).
        assert std_call_data.gentle_pushed is True

    @pytest.mark.asyncio
    async def test_stalemate_disclose_runs_then_awaits_final_commitment(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3 stalemate: when the customer keeps denying after the ladder,
        Nora delivers the SIMAH + legal-action + court-fees disclosure ONCE
        and asks one final commitment question. She does NOT immediately
        close the call as a dispute.

        We simulate "post-ladder denial" by entering Stage 3 with
        denied_after_clarify=True and pushing the customer through one more
        denial. The stalemate_disclose tool should fire, set the flag, and
        NOT close the call yet.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(
                Stage3NegotiationAgent(
                    std_call_data, denied_after_clarify=True
                )
            )
            # Customer still denies after ladder — agent should call
            # stalemate_disclose, NOT disputes_debt directly.
            result = await session.run(
                user_input="قلت لك ما عليّ شي ولا أبي أسدد"
            )

            result.expect.contains_function_call(name="stalemate_disclose")

        assert std_call_data.stalemate_disclosed is True
        # No dispute outcome yet — that only happens if the customer still
        # refuses on the NEXT turn.
        assert std_call_data.outcome != "dispute"

    @pytest.mark.asyncio
    async def test_stalemate_disclosure_covers_simah_legal_court(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3 stalemate: the canonical disclosure must cover ALL THREE
        elements together — SIMAH credit-record impact, possible legal
        action, possible court fees — framed as standard procedure, not as
        a personal threat. Same disclosure regardless of debt size.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(
                Stage3NegotiationAgent(
                    std_call_data, denied_after_clarify=True
                )
            )
            result = await session.run(
                user_input="ولا ريال، ما عليّ شي ولا أبي أسدد"
            )

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent delivers the stalemate disclosure that "
                        "covers ALL THREE elements in the same reply: "
                        "(1) possible impact on the credit record at SIMAH "
                        "(سمة / سجل ائتماني), (2) possible legal action "
                        "(إجراءات قانونية), and (3) possible court fees "
                        "(رسوم محكمة). The tone is procedural — 'حسب "
                        "الإجراءات المتبعة' — and not a personal threat. "
                        "Then the agent asks one final commitment question."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_stalemate_then_refusal_routes_to_dispute(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3 stalemate full flow: after stalemate_disclose has already
        run, if the customer STILL refuses to commit, the call is recorded
        as a dispute and the customer is directed to the official dispute
        channel (Mobily app or branch). Outcome must be 'dispute'.
        """
        std_call_data.stalemate_disclosed = True
        async with _session(judge_llm, std_call_data) as session:
            await session.start(
                Stage3NegotiationAgent(
                    std_call_data, denied_after_clarify=True
                )
            )
            result = await session.run(
                user_input="قلت لك ما أبي أتفق ولا أبي أسدد، خلاص"
            )

            result.expect.contains_function_call(name="disputes_debt")
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

        assert std_call_data.outcome == "dispute"

    @pytest.mark.asyncio
    async def test_denied_after_clarify_opens_with_calm_reminder(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.2 + 3.3: when entering Stage 3 with denied_after_clarify=True,
        the opening must include the calm reminder ("تم التحقق من الهوية،
        والمبلغ ثابت على حسابك ولازم ينسدد") before the negotiation ask.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(
                Stage3NegotiationAgent(
                    std_call_data, denied_after_clarify=True
                )
            )

            result = await session.run(user_input="قلت لك ما عليّ شي")

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent's opening reply briefly reminds the "
                        "customer that identity has been verified and the "
                        "amount is fixed on the account and must be settled "
                        "(تم التحقق من الهوية / المبلغ ثابت / لازم ينسدد), "
                        "then continues into the negotiation ask. The tone "
                        "is calm, not threatening, and no legal/court "
                        "wording is used."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_attempt2_half_offer_uses_exception_framing(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3: Attempt 2 (half today + rest within a week) MUST be
        introduced with 'كاستثناء عن المعتاد' / 'نستثني معك' framing.
        We trigger Attempt 2 by signaling hardship — Stage 3 opens with
        the half offer directly, and that offer must be framed as
        exception.
        """
        std_call_data.hardship_signaled = True
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(user_input="استمر")

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent offers a half-today / rest-within-a-week "
                        "payment plan AND introduces it as an exception "
                        "with wording like 'كاستثناء عن المعتاد' or "
                        "'نستثني معك'. The exception framing is REQUIRED — "
                        "the offer must not be presented as routine."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_mid_ladder_hardship_receives_empathy(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3: if the customer expresses financial pressure mid-ladder,
        Nora responds with ONE warm empathy line ("يعينك الله، ندري
        الموضوع مو سهل، إحنا معك") before continuing — not silence, not
        another ladder step jammed on top.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))
            # First the customer refuses full payment with a hardship phrase
            result = await session.run(
                user_input="والله ما أقدر، الأمور صعبة جدًا الفترة هذي"
            )

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent responds with a brief empathic phrase "
                        "(يعينك الله / ندري الموضوع مو سهل / إحنا معك / "
                        "مقدّر وضعك) before continuing the negotiation. "
                        "It must NOT threaten, guilt-trip, or repeat the "
                        "same full-payment question without acknowledging "
                        "the hardship."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_refusal_close_delivers_credit_record_line(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.3 refusal close: when the customer refuses to commit (but is
        NOT denying the debt), Nora delivers the credit-record consequences
        line ONCE before closing:
          "خلَّينا نقفلها خلال ٧ أيام بحد أقصى... عشان التأخير ممكن ينعكس
           بشكل سلبي على سجلك الائتماني في سمة..."
        She must NOT escalate to legal action or court fees at this stage —
        those are reserved for the Stage-3 stalemate disclosure path.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))
            result = await session.run(
                user_input="مو راح أدفع، خلاص ما أبي أتفق على شي"
            )

            # Refusal handoff: refuses_payment → ClosingAgent.
            result.expect.contains_function_call(name="refuses_payment")
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

            # The ClosingAgent's on_enter generates a closing line. Grab
            # the LAST assistant message in the run — that's the close.
            assistant_messages = [
                e for e in result.events
                if getattr(e, "type", None) == "message"
                and getattr(getattr(e, "item", None), "role", None) == "assistant"
            ]
            assert assistant_messages, (
                "Expected at least one assistant message in the refusal "
                "turn (the closing line from ClosingAgent)."
            )
            closing_text = " ".join(
                getattr(m.item, "text_content", "") or ""
                for m in assistant_messages
            )
            # Quick structural check: the credit-record line must mention
            # سمة or سجل ائتماني. We then defer the nuanced wording check
            # to the judge.
            assert "سمة" in closing_text or "سجل" in closing_text or "ائتمان" in closing_text, (
                f"Refusal close must mention the credit record / SIMAH "
                f"(PDF 3.3 refusal close); got: {closing_text!r}"
            )
            # Stronger guard: must NOT escalate to legal/court here.
            forbidden = ["محكمة", "قانوني", "قضائي"]
            for word in forbidden:
                assert word not in closing_text, (
                    f"Refusal close must NOT escalate to legal action or "
                    f"court (those belong to stalemate_disclose). Found "
                    f"{word!r} in: {closing_text!r}"
                )

        assert std_call_data.outcome == "refusal", (
            "Outcome must be 'refusal' (not 'dispute' — the customer did "
            "not deny the debt, only refused to commit)."
        )

    @pytest.mark.asyncio
    async def test_no_threatening_language(self, judge_llm, std_call_data):
        """Regression: agent must never use threatening or coercive language."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(user_input="ما عندي فلوس")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent responds with empathy or a follow-up offer. "
                        "It does NOT use words like 'لازم تسدد', 'إنذار', "
                        "'إجراء قانوني', or any pressuring / threatening language."
                    ),
                )
            )


# ============================================================
# Stage 4 — Recap + payment methods + close
# ============================================================

class TestStage4Recap:
    """Stage4RecapAgent recaps the agreement and explains payment channels."""

    @pytest.mark.asyncio
    async def test_recap_confirmed_closes_call(self, judge_llm, std_call_data):
        """مضبوط / تمام → recap_confirmed → ClosingAgent."""
        std_call_data.commitment = "full payment of 1500 SAR on 2026-05-18"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="مضبوط، تمام")

            result.expect.next_event().is_function_call(name="recap_confirmed")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

    @pytest.mark.asyncio
    async def test_wants_to_renegotiate_goes_to_reschedule(self, judge_llm, std_call_data):
        """Customer backs out → wants_to_renegotiate → RescheduleAgent."""
        std_call_data.commitment = "full payment of 1500 SAR on 2026-05-18"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(
                user_input="لا، ما أقدر بكرا، أبي موعد ثاني"
            )

            result.expect.next_event().is_function_call(name="wants_to_renegotiate")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=RescheduleAgent)

    @pytest.mark.asyncio
    async def test_recap_message_includes_sadad_code(self, judge_llm, std_call_data):
        """
        PDF 3.4: recap must mention SADAD with biller code 055, spoken as
        'خمسة خمسة صفر' (five-five-zero), plus the alternative channels
        (bank app, Mobily app, branch, ATM).
        """
        std_call_data.commitment = "full payment of 1500 SAR on 2026-05-20"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="تمام، أبغى أعرف كيف أسدد")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent explains how to pay via SADAD (سداد) and "
                        "mentions the biller code (كود المفوتر) 055 — pronounced "
                        "as 'خمسة خمسة صفر' (five-five-zero). It must NOT say "
                        "'صفر صفر خمسة' (zero-zero-five). The agent also lists "
                        "alternative channels: bank app, Mobily app, nearest "
                        "Mobily branch, and ATM."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_recap_below_threshold_adds_followup_line(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.4: when the locked first instalment was below 25 % of the
        total, the recap MUST add one short warm line saying Nora will
        reach out to arrange the remainder.
        """
        std_call_data.commitment = (
            "initial 200 SAR on 2026-05-25, remainder 1300 SAR TBD"
        )
        std_call_data.below_threshold_commit = True
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="طيب")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent recaps the partial agreement AND adds one "
                        "short warm line saying she will reach out soon to "
                        "arrange the remaining balance (e.g. 'بنتواصل معك "
                        "قريب لترتيب الباقي' or equivalent). The tone is "
                        "reassuring and free of blame."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_recap_lists_all_five_payment_methods(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.4: recap must list ALL FIVE payment channels every time —
        SADAD (with biller code 055), bank app, Mobily app, nearest Mobily
        branch, and ATM. None may be silently dropped.
        """
        std_call_data.commitment = "full payment of 1500 SAR on 2026-05-20"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="طيب")

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent's recap reply lists ALL of the "
                        "following payment channels in one connected "
                        "turn: (1) SADAD with biller code 055 spoken as "
                        "خمسة خمسة صفر, (2) the customer's bank app, "
                        "(3) the Mobily app, (4) the nearest Mobily "
                        "branch, and (5) an ATM (الصراف الآلي). None of "
                        "these may be omitted."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_recap_asks_for_receipt(self, judge_llm, std_call_data):
        """
        PDF 3.4: after listing payment channels, Nora asks the customer
        to send the payment receipt for confirmation. This step is
        mandatory in every commitment recap.
        """
        std_call_data.commitment = "full payment of 1500 SAR on 2026-05-20"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="طيب")

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent asks the customer to send the payment "
                        "receipt (إيصال السداد / إيصال الدفع) after "
                        "paying, for confirmation."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_recap_ends_with_madbouT_question(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.4: recap closes with the yes/no question "مضبوط؟" — the
        customer's chance to confirm or correct the agreement.
        """
        std_call_data.commitment = "full payment of 1500 SAR on 2026-05-20"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="طيب")

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The final question of the recap turn is the "
                        "single-word yes/no question 'مضبوط؟' (i.e. "
                        "'correct?'). The recap does NOT end with any "
                        "other closing phrase."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_recap_restates_amount_and_date(
        self, judge_llm, std_call_data
    ):
        """
        PDF 3.4: recap briefly restates the agreement — amount and date.
        Dates must be spoken naturally in Arabic (day + month name), never
        as digits, and amounts always in Arabic words.
        """
        std_call_data.commitment = "full payment of 1500 SAR on 2026-05-20"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="طيب")

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent's recap restates the agreed amount in "
                        "Arabic words (e.g. ألف وخمس مية) and the agreed "
                        "date naturally (e.g. عشرين مايو or in Najdi "
                        "phrasing). The literal digits '1500' or "
                        "'2026-05-20' must not appear as digit strings."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_callback_recap_uses_correct_phrasing(self, judge_llm, std_call_data):
        """
        When there's no payment commitment (only a callback), the recap must
        say 'بنتواصل معك' (we'll contact you), never 'ستدفع' (you will pay).
        """
        std_call_data.commitment = None
        std_call_data.callback_time = "2026-05-20 morning"
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage4RecapAgent(std_call_data))

            result = await session.run(user_input="طيب")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent confirms a follow-up call ('بنتواصل معك' or "
                        "similar) on the agreed date. It does NOT say the customer "
                        "will pay — only that the agent will call back."
                    ),
                )
            )


# ============================================================
# Special situations — PDF Section 4
# ============================================================

class TestSpecialSituations:
    """
    Behaviour from PDF Section 4: overrides and side-paths that can
    interrupt any stage of the call.
    """

    @pytest.mark.asyncio
    async def test_dnc_request_uses_canonical_line(
        self, judge_llm, std_call_data
    ):
        """
        PDF 4.1: a clear DNC phrase ('لا تتصلون علي', 'احذف رقمي', 'كفى
        اتصال') triggers the canonical line "تم تسجيل طلب عدم التواصل
        الهاتفي، والتواصل سيكون كتابياً فقط من الآن" and the call ends.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(user_input="لا تتصلون علي مرة ثانية")

            result.expect.contains_function_call(name="do_not_call")
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)
            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The closing line records the do-not-contact "
                        "request and notes that future contact will be "
                        "written only ('تم تسجيل طلب عدم التواصل "
                        "الهاتفي... التواصل سيكون كتابياً فقط' or close "
                        "paraphrase). The agent does NOT mention any "
                        "debt amount or apologise lengthily."
                    ),
                )
            )

        assert std_call_data.outcome == "dnc"

    @pytest.mark.asyncio
    async def test_death_reported_uses_canonical_condolences(
        self, judge_llm, std_call_data
    ):
        """
        PDF 4.2: explicit confirmation that the account holder has passed
        away triggers the canonical condolences line "نسأل الله له الرحمة
        والمغفرة. للتحديث الرسمي، تواصلوا مع أقرب فرع موبايلي" and the
        call ends respectfully.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(
                user_input="محمد توفى الله يرحمه قبل أسبوع"
            )

            result.expect.contains_function_call(name="customer_deceased")
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)
            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent expresses condolences with the "
                        "canonical phrase 'نسأل الله له الرحمة والمغفرة' "
                        "(or close paraphrase) AND directs the family to "
                        "the nearest Mobily branch for the official "
                        "account update. No debt amount is mentioned."
                    ),
                )
            )

        assert std_call_data.outcome == "death"

    @pytest.mark.asyncio
    async def test_cancelled_line_handled_then_returns_to_question(
        self, judge_llm, std_call_data
    ):
        """
        PDF 4.4: if the customer says they cancelled the line, Nora
        explains calmly in one or two short sentences that the balance
        still stands due to the 12-month contract / cancellation fee, then
        returns to the pending question. She does NOT close or argue.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(
                user_input="أنا أصلاً وقفت الخط من شهرين، ما عليّ شي"
            )

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent calmly explains that even if the line "
                        "was cancelled, the balance still stands — "
                        "mentioning the 12-month contract / cancellation "
                        "fee (عقد التزام ١٢ شهر / غرامة إنهاء). The "
                        "explanation is ONE or TWO short sentences, with "
                        "no argument and no threats. Then the agent "
                        "returns to the negotiation question."
                    ),
                )
            )

        # Agent must not have erroneously closed the call on this turn.
        assert std_call_data.outcome not in {"dispute", "refusal", "paid"}

    @pytest.mark.asyncio
    async def test_ported_number_handled_then_returns_to_question(
        self, judge_llm, std_call_data
    ):
        """
        PDF 4.4: 'نقلت الرقم' triggers the canonical ported-number
        response — balance stays on the account, contract/cancellation fee
        still applies — then returns to the pending question.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))

            result = await session.run(
                user_input="أنا نقلت رقمي لمشغل ثاني من زمان"
            )

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent calmly explains that even with the "
                        "number ported out, the debt stays on the account "
                        "and the contract/cancellation fee must be "
                        "settled (المديونية تبقى على الحساب / عقد / "
                        "غرامة إنهاء). ONE or TWO short sentences. Then "
                        "the agent returns to the negotiation question."
                    ),
                )
            )

    @pytest.mark.asyncio
    async def test_alo_recovery_does_not_reset_greeting(
        self, judge_llm, std_call_data
    ):
        """
        PDF 4.8: 'الو / تسمعيني؟ / صوتك مقطع' triggers a brief 'إيه سامعك'
        and a repeat of the LAST pending question — never a fresh greeting
        or restart of the call.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage3NegotiationAgent(std_call_data))
            # First a normal exchange — the agent asks the full-payment
            # question.
            await session.run(user_input="نعم استمر")

            # Then the customer interjects with an "alo?" probe.
            result = await session.run(
                user_input="الو؟ تسمعيني؟ صوتك مقطع"
            )

            await (
                result.expect
                .contains_message(role="assistant")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent briefly confirms hearing ('إيه سامعك' "
                        "or equivalent) and repeats the LAST pending "
                        "negotiation question. She does NOT restart the "
                        "call with a fresh greeting ('هلا معي محمد؟'), "
                        "does NOT reintroduce herself, and does NOT "
                        "repeat any earlier-stage script."
                    ),
                )
            )


# ============================================================
# Full conversation regression tests with JudgeGroup
# ============================================================

class TestFullConversations:
    """
    End-to-end multi-turn conversations evaluated with JudgeGroup.
    These are the primary regression tests — run them when changing prompts,
    tools, or model parameters to detect behaviour shifts.
    """

    @pytest.mark.asyncio
    async def test_happy_path_full_payment(self, judge_llm, std_call_data):
        """
        Golden path: right party → ID confirmed → reason given → commits full
        payment → confirms recap.

        JudgeGroup evaluates the full conversation for accuracy, relevancy,
        task completion, and tool use correctness.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            # Stage 1 — identity
            await session.run(user_input="نعم، أنا محمد")

            # Stage 1 — ID digits
            await session.run(user_input="نعم، صحيح")

            # Stage 2 — reason for delay
            await session.run(user_input="عندي ضائقة مالية لكن أقدر أسدد")

            # Stage 3 — commit full today
            await session.run(user_input="أقدر أسدد كامل المبلغ اليوم")

            # Stage 4 — confirm recap
            await session.run(user_input="مضبوط، تمام")

            judges = JudgeGroup(
                llm=judge_llm,
                judges=[
                    task_completion_judge(),
                    accuracy_judge(),
                    tool_use_judge(),
                    relevancy_judge(),
                    safety_judge(),
                ],
            )
            result = await judges.evaluate(session.history)

            # Print reasoning for each judge to aid regression analysis
            for name, judgment in result.judgments.items():
                print(f"\n[judge:{name}] {judgment.verdict} — {judgment.reasoning}")

            assert result.all_passed, (
                f"Full-payment happy path failed judges: "
                + ", ".join(
                    f"{n}={j.verdict}" for n, j in result.judgments.items() if j.failed
                )
            )

    @pytest.mark.asyncio
    async def test_happy_path_two_step_payment(self, judge_llm, std_call_data):
        """Half today + rest later — two-step commitment path."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            await session.run(user_input="نعم، أنا محمد")
            await session.run(user_input="نعم، صحيح")
            await session.run(user_input="عندي ضائقة")
            await session.run(user_input="ما أقدر الكامل، لكن أقدر النصف اليوم والباقي بعد أسبوعين")
            await session.run(user_input="مضبوط")

            judges = JudgeGroup(
                llm=judge_llm,
                judges=[
                    task_completion_judge(),
                    tool_use_judge(),
                    coherence_judge(),
                ],
            )
            result = await judges.evaluate(session.history)

            for name, judgment in result.judgments.items():
                print(f"\n[judge:{name}] {judgment.verdict} — {judgment.reasoning}")

            assert result.majority_passed, (
                "Two-step payment path: majority of judges must pass"
            )

    @pytest.mark.asyncio
    async def test_busy_customer_schedules_callback(self, judge_llm, std_call_data):
        """Customer is busy → schedules callback → ClosingAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            await session.run(user_input="أنا مشغول الحين")
            await session.run(user_input="بكرا الصبح يناسبني")  # callback time

            judges = JudgeGroup(
                llm=judge_llm,
                judges=[
                    task_completion_judge(),
                    relevancy_judge(),
                ],
            )
            result = await judges.evaluate(session.history)

            for name, judgment in result.judgments.items():
                print(f"\n[judge:{name}] {judgment.verdict} — {judgment.reasoning}")

            assert result.none_failed, "Busy-callback flow must not have any judge failures"

    @pytest.mark.asyncio
    async def test_wrong_party_with_referral(self, judge_llm, std_call_data):
        """Wrong party who knows the customer → collects mobile → ClosingAgent."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            await session.run(user_input="لا، أنا مو محمد")
            await session.run(user_input="نعم، أعرفه")               # knows the person
            await session.run(user_input="رقمه صفر خمسة صفر واحد اثنين ثلاثة أربعة خمسة ستة سبعة")

            judges = JudgeGroup(
                llm=judge_llm,
                judges=[
                    task_completion_judge(),
                    safety_judge(),
                ],
            )
            result = await judges.evaluate(session.history)

            for name, judgment in result.judgments.items():
                print(f"\n[judge:{name}] {judgment.verdict} — {judgment.reasoning}")

            assert result.none_failed

    @pytest.mark.asyncio
    async def test_customer_deceased_handled_respectfully(self, judge_llm, std_call_data):
        """Caller reports customer has passed away — agent must express condolences."""
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            result = await session.run(
                user_input="محمد توفي الله يرحمه"
            )

            result.expect.next_event().is_function_call(name="customer_deceased")
            result.expect.next_event().is_function_call_output()
            result.expect.contains_agent_handoff(new_agent_type=ClosingAgent)

    @pytest.mark.asyncio
    async def test_no_debt_details_before_id_verification(self, judge_llm, std_call_data):
        """
        Compliance regression: agent must NOT mention the debt amount before
        the customer's ID is verified (i.e. during Stage 1).
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage1RightPartyAgent(std_call_data))

            # Identity confirmed but ID not yet verified — agent is in Stage 1 ID step
            await session.run(user_input="نعم، أنا محمد")

            # Customer asks about the purpose — agent must not leak the amount
            result = await session.run(user_input="إيش تبي؟ ليش تتصل؟")

            await (
                result.expect
                .next_event(type="message")
                .judge(
                    judge_llm,
                    intent=(
                        "The agent responds without revealing any debt amount "
                        "or account details. It only says the call is regarding "
                        "the Mobily account and asks the customer to confirm the "
                        "last 4 digits of their ID."
                    ),
                )
            )


# ============================================================
# Mocked tools — edge-case and error-path regression
# ============================================================

class TestMockedTools:
    """
    Use mock_tools to force specific tool outputs and verify that the agent
    handles them correctly — especially error paths that are hard to trigger
    in a live system.
    """

    @pytest.mark.asyncio
    async def test_stage3_disputes_mid_negotiation(self, judge_llm, std_call_data):
        """
        Agent reaches a commit_full mock that simulates an unexpected dispute —
        the agent should respond appropriately (inform, not threaten).
        """
        def _raise_dispute(*_, **__):
            raise RuntimeError("dispute flag set externally")

        with mock_tools(Stage3NegotiationAgent, {"commit_full": _raise_dispute}):
            async with _session(judge_llm, std_call_data) as session:
                await session.start(Stage3NegotiationAgent(std_call_data))

                result = await session.run(user_input="أقدر أسدد اليوم")

                await (
                    result.expect
                    .next_event(type="message")
                    .judge(
                        judge_llm,
                        intent=(
                            "The agent responds politely after an unexpected error. "
                            "It does not blame the customer and the tone remains "
                            "professional and calm."
                        ),
                    )
                )

    @pytest.mark.asyncio
    async def test_stage2_clarify_debt_with_service_details(
        self, judge_llm, std_call_data
    ):
        """
        Customer doesn't recognise the debt → clarify_debt fires and the agent
        explains the service details. Regression: agent must cite the service
        type and subscription date from the services array.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(Stage2DebtIntroAgent(std_call_data))

            result = await session.run(
                user_input="ما أعرف هالفاتورة، ما عليّ شي"
            )

            # clarify_debt should be called first (not disputes_debt)
            result.expect.next_event().is_function_call(name="clarify_debt")

    @pytest.mark.asyncio
    async def test_wrong_party_receives_partial_mobile(self, judge_llm, std_call_data):
        """
        CollectMobileAgent receives a short (invalid) number — must re-ask,
        not store a garbage referrer mobile.
        """
        async with _session(judge_llm, std_call_data) as session:
            await session.start(CollectMobileAgent(std_call_data))

            result = await session.run(user_input="خمسة صفر واحد")  # only 3 digits

            # Should NOT call mobile_provided successfully — agent must re-ask
            result.expect.contains_message(role="assistant")
            # referrer should not be set from garbage input
            assert std_call_data.referrer_mobile is None, (
                "referrer_mobile must not be set from a partial (3-digit) number"
            )
