"""Trivia quiz agent built with PydanticAI.

Usage (from the project root):
    uv run python src/trivia_agent.py
    # or: python src/trivia_agent.py

Requires OPENAI_API_KEY in the environment.
Logfire tracing is enabled automatically if LOGFIRE_TOKEN is set
(or after `logfire auth` + `logfire projects use <name>`); otherwise
the agent still runs and nothing is sent.
"""

import logfire
from pydantic_ai import Agent

from trivia_tools import TriviaTools


# ── Monitoring (used in the later questions) ─────────────────────────
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


# ── Instructions ─────────────────────────────────────────────────────
instructions = """You are a trivia quizmaster. When asked to play trivia:
1. Use the available tools to fetch trivia questions
2. Ask the player one question at a time with multiple choice options
3. Wait for their answer before moving to the next question
4. When the player answers, explain why the correct answer is correct - add interesting context and facts
5. After all questions, give the final score

Additional rules:
- If the player names a category (e.g. "History"), call get_categories first
  to find its numeric ID, then call get_questions with that ID.
- The tool output tells you which answer is correct. NEVER reveal it before
  the player answers. Shuffle the options and label them A, B, C, D so the
  correct one is not always in the same position.
- Keep track of the score and show it after each answer (e.g. "Score: 2/3").
"""


# ── Agent ────────────────────────────────────────────────────────────
trivia_tools = TriviaTools()

agent = Agent(
    "openai:gpt-4o-mini",
    name="agent",
    tools=[trivia_tools.get_categories, trivia_tools.get_questions],
    instructions=instructions,
)


# ── Interactive loop ─────────────────────────────────────────────────
message_history = []


def run(user_input: str) -> str:
    """Send one message to the agent, keeping the conversation history."""
    global message_history
    result = agent.run_sync(user_input, message_history=message_history)
    # Keep the full conversation so the agent remembers the questions,
    # the correct answers and the score between turns.
    message_history = result.all_messages()
    return result.output


def main() -> None:
    print("🎲 Trivia Quiz — type 'quit' to exit\n")

    # One parent span for the whole session: every agent run (one per turn)
    # is nested under it, so the full quiz shows up as a single trace.
    with logfire.span("trivia_session"):
        # Start the session with the homework prompt
        first_message = "Let's play 5 easy questions from Science & Nature"
        print(f"You: {first_message}")
        print(f"\nQuizmaster: {run(first_message)}\n")

        # Then answer the questions interactively
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "stop"}:
                print("Bye! 👋")
                break

            answer = run(user_input)
            print(f"\nQuizmaster: {answer}\n")


if __name__ == "__main__":
    main()
