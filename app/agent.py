"""Step 3: the agent. A LangGraph loop where the LLM decides which tool to call,
reads the tool result, and repeats until it can answer.

    START -> agent --(wants a tool?)--> tools -> agent -> ... -> END

Run:  python -m app.agent "Is http://paypa1.com/login safe?"
      python -m app.agent            (interactive chat)
"""
import sys

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app import config  # noqa: F401  (loads .env and quiet settings first)
from app.llm import get_llm
from app.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a security threat-intelligence assistant for security analysts.
- Use search_threat_intel for questions about vulnerabilities, attacker techniques,
  detection or remediation. Use lookup_cve when the user gives an exact CVE ID.
  Use check_url when the user asks whether a link or domain is safe.
- Specific facts (affected or fixed version numbers, dates, configuration settings,
  CVE and technique details) must come from tool results. Cite them in square
  brackets, e.g. [CVE-2021-44228] or [T1566]. Never state a version number or
  setting that is not in the tool results; if the tools do not contain it, say so.
- You may add widely accepted security best practices, but put them under a
  heading "General guidance (not from the knowledge base)".
- Be concise: a short answer first, then recommended actions as bullet points."""

MAX_STEPS = 12  # safety limit on agent <-> tool round trips


def build_agent():
    llm_with_tools = get_llm().bind_tools(ALL_TOOLS)

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call -> "tools", else END
    graph.add_edge("tools", "agent")
    return graph.compile()


def ask(agent, question: str) -> dict:
    """Run one question; return the answer and the tools that were used."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": MAX_STEPS},
    )
    tools_used = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    return {"answer": result["messages"][-1].content, "tools_used": tools_used}


def main() -> None:
    agent = build_agent()
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else None
    if questions is None:
        print("Security Threat Intel Agent. Type a question, or 'exit' to quit.")
    while True:
        question = questions.pop() if questions else input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"} or not question:
            break
        out = ask(agent, question)
        print(f"\n[tools used: {', '.join(out['tools_used']) or 'none'}]\n")
        print(out["answer"])
        if questions is not None and not questions:
            break


if __name__ == "__main__":
    main()
