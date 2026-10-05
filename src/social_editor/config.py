import os

from crewai import LLM

DEFAULT_MODEL = "gemini/gemini-3.8-flash"

# Key variables for the common hosted providers. Others (e.g. local Ollama) are passed through.
API_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
}


def model_name() -> str:
    return os.environ.get("MODEL") or DEFAULT_MODEL


def missing_key_error(model: str) -> str | None:
    """Return a user-facing message if the model's provider needs a key that isn't set."""
    names = API_KEYS.get(model.split("/", 1)[0], ())
    if not names or any(os.environ.get(name) for name in names):
        return None
    return f"`{model}` needs an API key. Set {' or '.join(names)} in your environment or `.env`."


def make_llm(model: str) -> LLM:
    return LLM(model=model, temperature=0.7)
