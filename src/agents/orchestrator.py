"""LangGraph workflow for task routing and agent coordination."""

from typing import Any, Literal

from langgraph.graph import END, StateGraph
from loguru import logger
from openai import OpenAI
from typing_extensions import TypedDict

from config.settings import settings
from src.agents.qa_agent import QAAgent
from src.agents.risk_agent import RiskAgent
from src.agents.summary_agent import SummaryAgent
from src.retrieval.retriever import Retriever


class AgentState(TypedDict, total=False):
    """State passed through the LangGraph workflow."""

    task_type: str  # "risk", "qa", "summary"
    question: str
    document_id: str
    context: str
    page_numbers: list[int]
    result: dict[str, Any]
    agent: str
    error: str | None


def classify_task(state: AgentState) -> AgentState:
    """Classify the incoming task and route to the correct agent."""
    task_type = state.get("task_type", "")

    if task_type in ("risk", "qa", "summary"):
        logger.info(f"Task classified as: {task_type}")
        return state

    # Auto-classify from question
    question = state.get("question", "").lower()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's intent about a legal contract into one of: "
                    "risk, qa, summary. Respond with only the single word."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=10,
    )

    classified = (response.choices[0].message.content or "qa").strip().lower()
    if classified not in ("risk", "qa", "summary"):
        classified = "qa"

    logger.info(f"Auto-classified task as: {classified}")
    return {**state, "task_type": classified}


def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant document chunks for the task."""
    question = state.get("question", "")
    task_type = state.get("task_type", "qa")

    if state.get("context"):
        return state

    # Build query based on task type
    if task_type == "risk":
        query = question or "What are the potential risks, liabilities, and problematic clauses?"
    elif task_type == "summary":
        query = question or "What are the key terms, parties, and obligations?"
    else:
        query = question

    retriever = Retriever(use_reranking=True, use_query_expansion=False, use_query_translation=True)
    result = retriever.retrieve(query, k=settings.RETRIEVAL_TOP_K)

    context_parts = []
    page_numbers = []
    for r in result.results:
        # Use parent context if available for richer context
        parent_id = r.metadata.get("parent_id")
        if parent_id and parent_id in result.parent_context:
            context_parts.append(result.parent_context[parent_id])
        else:
            context_parts.append(r.text)

        page = r.metadata.get("page")
        if page and page not in page_numbers:
            page_numbers.append(page)

    context = "\n\n---\n\n".join(context_parts)

    logger.info(f"Retrieved {len(result.results)} chunks for {task_type} task")
    return {**state, "context": context, "page_numbers": page_numbers}


def route_to_agent(state: AgentState) -> Literal["risk_agent", "qa_agent", "summary_agent"]:
    """Route to the appropriate agent based on task type."""
    task_type = state.get("task_type", "qa")
    route_map = {
        "risk": "risk_agent",
        "qa": "qa_agent",
        "summary": "summary_agent",
    }
    return route_map.get(task_type, "qa_agent")


# Initialize agents
_risk_agent = RiskAgent()
_qa_agent = QAAgent()
_summary_agent = SummaryAgent()


def run_risk_agent(state: AgentState) -> AgentState:
    return _risk_agent(state)


def run_qa_agent(state: AgentState) -> AgentState:
    return _qa_agent(state)


def run_summary_agent(state: AgentState) -> AgentState:
    return _summary_agent(state)


def build_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_task)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("risk_agent", run_risk_agent)
    graph.add_node("qa_agent", run_qa_agent)
    graph.add_node("summary_agent", run_summary_agent)

    # Define edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_conditional_edges("retrieve", route_to_agent)
    graph.add_edge("risk_agent", END)
    graph.add_edge("qa_agent", END)
    graph.add_edge("summary_agent", END)

    return graph


# Compiled graph (singleton)
_compiled_graph = None


def get_orchestrator():
    """Get the compiled orchestrator graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


def run(
    task_type: str = "",
    question: str = "",
    document_id: str = "",
    context: str = "",
) -> dict[str, Any]:
    """Run the orchestrator with the given inputs.

    Args:
        task_type: "risk", "qa", or "summary". Auto-classified if empty.
        question: User question (for qa) or empty (for risk/summary).
        document_id: ID of ingested document.
        context: Pre-retrieved context (skips retrieval if provided).

    Returns:
        Final state dict with 'result' key containing agent output.
    """
    orchestrator = get_orchestrator()

    initial_state: AgentState = {
        "task_type": task_type,
        "question": question,
        "document_id": document_id,
        "context": context,
        "page_numbers": [],
        "result": {},
        "agent": "",
        "error": None,
    }

    logger.info(f"Orchestrator running: task_type={task_type or 'auto'}")
    final_state = orchestrator.invoke(initial_state)
    logger.info(f"Orchestrator complete: agent={final_state.get('agent')}")

    return final_state
