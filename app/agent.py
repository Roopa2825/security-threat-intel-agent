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

Tools:
- search_threat_intel: questions about vulnerabilities, attacker techniques, detection or remediation.
- lookup_cve: when the user gives an exact CVE ID.
- check_url: when the user asks whether a link or domain is safe.

Answer format:
1. One or two sentences that directly answer the question.
2. "From the knowledge base": bullet points with specific facts taken from the tool
   results. End every bullet with its source in square brackets, e.g. [CVE-2021-44228],
   [T1566] or [check_url]. Copy version numbers, dates and settings exactly as the tool
   results give them. If a detail is not in the tool results, leave it out of this section.
3. "General guidance (not from the knowledge base)" (optional): widely accepted best
   practices only. No version numbers, configuration settings, dates or CVE details here.

If the tools do not contain the answer, say so instead of guessing."""

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
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    return {
        "answer": result["messages"][-1].content,
        "tools_used": [m.name for m in tool_messages],
        "tool_outputs": [str(m.content) for m in tool_messages],
    }


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
