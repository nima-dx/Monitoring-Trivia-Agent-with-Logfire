"""Reconstruct a grouped trivia session from Logfire and calculate its cost.

Usage (from the project root):
    export LOGFIRE_READ_TOKEN=pylf_v1_eu_...
    uv run python src/session_cost.py                 # newest trivia_session
    uv run python src/session_cost.py <trace_id>      # a specific session
"""

import os
import sys
from datetime import datetime, timedelta, timezone

from logfire.query_client import LogfireQueryClient

from trace_replay import fetch_trace, trace_to_run_result

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def calculate_cost(model_name, input_tokens, output_tokens):
    prices = MODEL_PRICES[model_name.lower()]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def latest_session_trace_id(client: LogfireQueryClient) -> str:
    """Trace ID of the most recent trivia_session span."""
    rows = client.query_json_rows(
        sql="""
        SELECT trace_id
        FROM records
        WHERE span_name = 'trivia_session'
        ORDER BY start_timestamp DESC
        LIMIT 1
        """,
        min_timestamp=datetime.now(timezone.utc) - timedelta(days=7),
    )["rows"]
    if not rows:
        raise SystemExit("No trivia_session span found in the last 7 days.")
    return rows[0]["trace_id"]


def main() -> None:
    with LogfireQueryClient(read_token=os.environ["LOGFIRE_READ_TOKEN"]) as client:
        trace_id = sys.argv[1] if len(sys.argv) > 1 else latest_session_trace_id(client)

        # Reads every invoke_agent span in the trace and sums their tokens
        trace = fetch_trace(trace_id, client)

    result = trace_to_run_result(trace)
    # `usage` is a method in older PydanticAI versions and a property in newer ones
    usage = result.usage
    if not hasattr(usage, "input_tokens"):
        usage = usage()

    cost = calculate_cost("gpt-4o-mini", usage.input_tokens, usage.output_tokens)

    print(f"trace_id:       {trace_id}")
    print(f"agent runs:     {trace.agent_runs}")
    print(f"messages:       {len(result.all_messages())}")
    print(f"input tokens:   {usage.input_tokens:,}")
    print(f"output tokens:  {usage.output_tokens:,}")
    print(f"total cost:     ${cost:.6f}")
    print(f"\nlast answer:\n{result.output}")


if __name__ == "__main__":
    main()
