import json
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from gemini_realtime_agent import build_instructions
from metrics_logger import setup_metrics

logger = logging.getLogger("agent-gemini-pipeline")

load_dotenv()


class DebtCollectorAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)

    async def on_enter(self):
        await self.session.generate_reply(allow_interruptions=True)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="debt-collector-gemini-pipeline")
async def entrypoint(ctx: JobContext):
    try:
        meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    except json.JSONDecodeError:
        logger.warning("could not parse job metadata; using empty dict")
        meta = {}

    session = AgentSession(
        stt=google.STT(
            model="latest_long",
            languages=["ar-SA"],
        ),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=google.TTS(
            voice_name="ar-XA-Wavenet-A",
            language="ar-XA",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption={"mode": "vad"},
            preemptive_generation={"enabled": True},
        ),
        vad=ctx.proc.userdata["vad"],
    )

    setup_metrics(session, ctx)

    await session.start(
        agent=DebtCollectorAgent(build_instructions(meta)),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
