# pyrefly: ignore [missing-import]
from openai import OpenAI
from langsmith.wrappers import wrap_openai

from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
)


client = wrap_openai(
    OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
)


# ==================================================
# ASK LLM
# ==================================================

def ask_llm(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the
provided research paper context.

If the context does not contain enough
information, clearly say that the information
is not available in the provided context.

Do not invent information.

USER QUESTION:
{question}

RESEARCH PAPER CONTEXT:
{context}

INSTRUCTIONS:

1. Answer in simple and clear language.
2. Explain technical concepts when necessary.
3. Use only the provided context.
4. Do not make up facts.
5. Give a useful answer even if the question
   asks for a simple explanation.
"""


    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful research "
                        "paper assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.2,
        )

        # ------------------------------------------
        # CHECK RESPONSE
        # ------------------------------------------

        if not response.choices:

            raise RuntimeError(
                "OpenRouter returned no choices."
            )

        message = response.choices[0].message

        content = message.content

        if content is None:

            raise RuntimeError(
                "LLM returned empty content."
            )

        content = content.strip()

        if not content:

            raise RuntimeError(
                "LLM returned an empty answer."
            )

        return content

    except Exception as e:

        print(
            f"\nLLM ERROR: {e}"
        )

        return (
            "I could retrieve relevant sections "
            "from the research paper, but the LLM "
            "failed to generate the answer.\n\n"
            f"Error: {e}"
        )