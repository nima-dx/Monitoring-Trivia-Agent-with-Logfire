# Monitoring-Trivia-Agent-with-Logfire

In this homework we'll build a trivia quiz agent, instrument it with monitoring, and analyze its behavior through traces. We provide the tools code - your focus is on the agent, monitoring, querying, and analysis.

We recommend using PydanticAI and Logfire, but you can use any agent framework and any observability platform - even one you built yourself - to answer these questions.

The agent runs a trivia quiz powered by the Open Trivia Database - a free API for trivia questions, no authentication needed. You can explore the available categories on the Open Trivia DB API config page.

Here's how a session works:

    You pick a category and say "Let's play 3 medium questions from History"
    The agent fetches questions from the API and asks you the first one with multiple choice options
    You answer, and the agent tells you if you're right or wrong, then asks the next question
    After all questions, the agent gives your final score and explains the correct answers
