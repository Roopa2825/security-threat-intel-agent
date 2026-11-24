"""Step 3: the agent. A LangGraph loop where the LLM decides which tool to call,
reads the tool result, and repeats until it can answer.

    START -> agent --(wants a tool?)--> tools -> agent -> ... -> END

Run:  python -m app.agent "Is http://paypa1.com/login safe?"
      python -m app.agent            (interactive chat)
"""
import sys
import time

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
- check_url: when the user asks whether a link or domain is safe. Pass the URL exactly as
  the user wrote it, character for character. Never fix spelling or digits in a URL:
  look-alike characters (e.g. 0 for o, 1 for l) are often the phishing signal itself.
- classify_phishing_text: when the user pastes the text of an email or message. If the
  message also contains links, check those links with check_url too.

Answer format:
1. One or two sentences that directly answer the question, using only facts from the
   tool results.
2. "From the knowledge base": bullet points with specific facts from the tool results.
   End every bullet with the exact source ID from the tool result in plain square
   brackets. For vulnerability and attacker-technique facts, cite the record ID shown in
   the tool result, e.g. [CVE-2021-44228] or [T1110], never the tool name: write
   [CVE-2021-44228], not [lookup_cve] or [search_threat_intel]. Only URL-check and
   email-classification results have no record ID; cite those as [check_url] or
   [classify_phishing_text]. Copy version numbers, dates, settings and URLs exactly.
   If a detail is not in the tool results, leave it out.
3. "General guidance (not from the knowledge base)" (optional): widely accepted best
   practices only. No version numbers, configuration settings, dates or CVE details here.

If the tools do not contain the answer, say so instead of guessing. For detection or
mitigation questions, if the tool results contain little or no detection or mitigation
guidance (for example, activity that happens outside the victim's network), say that
plainly in the opening sentences, and put any general ideas only under "General guidance"."""

MAX_STEPS = 12  # safety limit on agent <-> tool round trips
RETRIES = 2  # extra attempts if the LLM provider returns an error
RETRY_WAIT_SECONDS = 5


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
    for attempt in range(1, RETRIES + 2):
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": MAX_STEPS},
            )
            break
        except Exception:  # e.g. rate limit or a malformed tool call from the LLM
            if attempt > RETRIES:
                raise
            time.sleep(RETRY_WAIT_SECONDS * attempt)
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
