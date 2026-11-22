"""Create the chat model. The agent works with several providers; pick one in .env:

  LLM_PROVIDER=groq     (needs GROQ_API_KEY, free tier)
  LLM_PROVIDER=gemini   (needs GOOGLE_API_KEY, free tier)
  LLM_PROVIDER=ollama   (runs locally, no key; install Ollama first)
"""
import os

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1:8b",
}


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, gemini or ollama.")
