import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100) * (len(values) - 1)))))
    return values[k]


def analyze(path: Path) -> None:
    by_type: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_type[rec.get("type", "Unknown")].append(rec)

    print(f"\n=== {path.name} ===")
    for t, records in sorted(by_type.items()):
        print(f"\n[{t}]  events={len(records)}")
        latency_fields = [
            "ttft",
            "ttfb",
            "duration",
            "end_of_utterance_delay",
            "transcription_delay",
            "on_user_turn_completed_delay",
        ]
        for field in latency_fields:
            values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
            if not values:
                continue
            print(
                f"  {field:>32s}: "
                f"n={len(values):3d}  "
                f"mean={statistics.mean(values):.3f}s  "
                f"p50={_pct(values, 50):.3f}s  "
                f"p95={_pct(values, 95):.3f}s  "
                f"max={max(values):.3f}s"
            )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python analyze_metrics.py <metrics-file.jsonl | metrics-dir>")
        return 1

    target = Path(argv[1])
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
        if not files:
            print(f"no .jsonl files in {target}")
            return 1
        for f in files:
            analyze(f)
    elif target.is_file():
        analyze(target)
    else:
        print(f"not found: {target}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
