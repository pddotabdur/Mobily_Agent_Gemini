import json
import logging
import time
from pathlib import Path

from livekit.agents import (
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    metrics,
)

logger = logging.getLogger("metrics-logger")


def setup_metrics(
    session: AgentSession,
    ctx: JobContext,
    output_dir: str = "metrics",
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"{ctx.room.name}_{int(time.time())}.jsonl"

    usage_collector = metrics.UsageCollector()

    def _serialize(obj) -> dict:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return {"repr": repr(obj)}

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        try:
            record = _serialize(ev.metrics)
            record["type"] = ev.metrics.__class__.__name__
            record["wall_ts"] = time.time()
            with output_file.open("a") as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"failed to write metric: {e}")

    async def _on_shutdown():
        summary = usage_collector.get_summary()
        logger.info(f"final usage for {ctx.room.name}: {summary}")
        try:
            record = _serialize(summary)
            record["type"] = "Summary"
            record["wall_ts"] = time.time()
            with output_file.open("a") as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"failed to write summary: {e}")

    ctx.add_shutdown_callback(_on_shutdown)
    logger.info(f"metrics will be written to {output_file}")
    return output_file
