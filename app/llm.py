"""Create the chat model. The agent works with several providers; pick one in .env:

  LLM_PROVIDER=groq     (needs GROQ_API_KEY, free tier)
  LLM_PROVIDER=gemini   (needs GOOGLE_API_KEY, free tier)
  LLM_PROVIDER=ollama   (runs locally, no key; install Ollama first)
  LLM_PROVIDER=anthropic (needs ANTHROPIC_API_KEY, paid; Claude models)
  LLM_PROVIDER=openai   (needs OPENAI_API_KEY, paid; GPT models)

Set LLM_MODEL to choose a specific model (recommended for Anthropic and OpenAI,
whose model names change over time). Providers retire models often, so for
Groq, if LLM_MODEL is not set, the code asks Groq which models are available
and picks the first suitable one.

See available Groq models:  python -m app.llm
"""
import os

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1:8b",
    # Small, low-cost models. If one is retired, set LLM_MODEL in .env.
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
}
# Preferred Groq chat models, best first. Only models Groq currently offers are used.
GROQ_PREFERRED = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
# Groq also hosts non-chat models (speech, safety filters); never pick these.
GROQ_EXCLUDE = ("whisper", "tts", "guard", "prompt-guard", "orpheus", "compound")


def list_groq_models() -> list[str]:
    from groq import Groq

    return sorted(m.id for m in Groq().models.list().data)


def pick_groq_model() -> str:
    available = list_groq_models()
    for name in GROQ_PREFERRED:
        if name in available:
            return name
    chat_models = [m for m in available if not any(x in m.lower() for x in GROQ_EXCLUDE)]
    if not chat_models:
        raise RuntimeError(f"No suitable Groq chat model found. Available: {available}")
    return chat_models[0]


def get_llm(model: str | None = None):
    """model overrides LLM_MODEL, e.g. to give the evaluation judge its own model (and quota)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = model or os.getenv("LLM_MODEL")
    if provider == "groq":
        from langchain_groq import ChatGroq

        model = model or pick_groq_model()
        print(f"[using Groq model: {model}]")
        return ChatGroq(model=model, temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or DEFAULT_MODELS["gemini"], temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or DEFAULT_MODELS["ollama"], temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = model or DEFAULT_MODELS["anthropic"]
        print(f"[using Anthropic model: {model}]")
        return ChatAnthropic(model=model, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = model or DEFAULT_MODELS["openai"]
        print(f"[using OpenAI model: {model}]")
        return ChatOpenAI(model=model, temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, gemini, ollama, anthropic or openai.")


if __name__ == "__main__":
    from app import config  # noqa: F401  (loads .env)

    print("Models available on your Groq account:")
    for model_id in list_groq_models():
        print(" ", model_id)
    print(f"\nThe agent would pick: {pick_groq_model()}")
