"""Query trivia session traces from Logfire.

Usage (from the project root):
    export LOGFIRE_READ_TOKEN=pylf_v1_eu_...      # read token, NOT the write token
    uv run python src/query_traces.py
"""

import os
from datetime import datetime, timedelta, timezone

from logfire.query_client import LogfireQueryClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

read_token = os.environ["LOGFIRE_READ_TOKEN"]

# Only look at the last 7 days (queries without a time window can be slow)
min_timestamp = datetime.now(timezone.utc) - timedelta(days=7)


# 1) One row per grouped session: the trivia_session parent spans
sessions_sql = """
SELECT
    trace_id,
    start_timestamp,
    end_timestamp,
    duration
FROM records
WHERE span_name = 'trivia_session'
ORDER BY start_timestamp DESC
"""

# 2) Per-trace summary to cross-check with the UI:
#    how many agent runs and how many input tokens per trace
summary_sql = """
SELECT
    trace_id,
    COUNT(*) FILTER (WHERE span_name = 'invoke_agent agent') AS agent_runs,
    SUM(CAST(attributes->>'gen_ai.usage.input_tokens' AS INT))
        FILTER (WHERE span_name LIKE 'chat %')               AS input_tokens,
    MIN(start_timestamp)                                     AS started
FROM records
GROUP BY trace_id
ORDER BY started DESC
"""

with LogfireQueryClient(read_token=read_token) as client:
    sessions = client.query_json_rows(sql=sessions_sql, min_timestamp=min_timestamp)
    summary = client.query_json_rows(sql=summary_sql, min_timestamp=min_timestamp)

print("=== trivia_session traces ===")
for row in sessions["rows"]:
    print(f"{row['start_timestamp']}  trace_id={row['trace_id']}  "
          f"duration={row['duration']:.1f}s")

print("\n=== all traces (agent runs / input tokens) ===")
for row in summary["rows"]:
    print(f"{row['started']}  trace_id={row['trace_id']}  "
          f"agent_runs={row['agent_runs']}  input_tokens={row['input_tokens']}")
