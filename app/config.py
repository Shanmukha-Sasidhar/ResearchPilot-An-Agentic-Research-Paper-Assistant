import os

from dotenv import load_dotenv

load_dotenv()


# ==================================================
# OPENROUTER
# ==================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)

LLM_MODEL = "openrouter/free"


# ==================================================
# LANGSMITH
# ==================================================

LANGSMITH_API_KEY = os.getenv(
    "LANGSMITH_API_KEY"
)

LANGSMITH_TRACING = os.getenv(
    "LANGSMITH_TRACING",
    "false"
)

LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "research-paper-agent"
)