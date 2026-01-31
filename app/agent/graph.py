"""
LangGraph agent: query rewrite → retrieve → analyze → (loop or generate).

Orchestration only; heavy compute via HF API. Max 2 retrieval passes.
Agentic mode: tool-calling loop (search_documents, calculator, web_search, etc.) via OpenAI.
"""

import json
import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.llm import chat_with_tools, hf_llm
from app.agent.tools import AGENT_TOOLS, execute_tool
from app.services.retrieval_service import retrieve_context

logger = logging.getLogger(__name__)
MAX_ITERATIONS = 2
MAX_AGENTIC_ROUNDS = 12


class AgentState(TypedDict):
    original_query: str
    rewritten_query: str
    retrieved_chunks: list
    answer: str
    needs_more_context: bool
    iteration: int
    chat_history: list  # list of {"role": "user"|"assistant", "content": str}


def _format_history(history: list, max_messages: int = 6) -> str:
    """Format last N messages for inclusion in prompts."""
    if not history:
        return ""
    recent = history[-max_messages:] if len(history) > max_messages else history
    lines = []
    for m in recent:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    if not lines:
        return ""
    return "Recent conversation:\n" + "\n".join(lines) + "\n\n"


def _rewrite_query(state: AgentState) -> dict:
    """Node 1: Rewrite query for better retrieval (HF LLM). History-aware."""
    query = state.get("original_query") or ""
    history = state.get("chat_history") or []
    logger.info("[graph:rewrite_query] IN  original_query=%r", query)
    logger.info("[graph:rewrite_query] IN  history_len=%d history_preview=%s", len(history), [m.get("role") + ": " + (m.get("content") or "")[:60] for m in history[-3:]])
    history_block = _format_history(history)
    prompt = f"""
        Convert the user query into a dense semantic search phrase.

        Rules:
        - Remove filler words (how, what, why, do, can, I, we, you)
        - Remove conversational language
        - Use noun-heavy keywords
        - If there is recent conversation, use it to resolve references (e.g. "that policy", "it" -> concrete topic)
        - Output 3–8 words only
        - No full sentences
        - No punctuation except spaces

        {history_block}Current query: {query}
        """
    logger.info("[graph:rewrite_query] prompt_len=%d", len(prompt))
    rewritten = hf_llm(prompt, max_new_tokens=100).strip() or query
    logger.info("[graph:rewrite_query] llm_raw=%r", rewritten)
    # Fallback: if rewrite looks like boolean syntax, use original query for retrieval
    if " OR " in rewritten or ("(" in rewritten and ")" in rewritten):
        rewritten = query
        logger.info("[graph:rewrite_query] fallback to original_query (rewrite had boolean syntax)")
    # Strip meta-commentary: use only first line/block (before "\n\n"); strip outer quotes
    if "\n\n" in rewritten:
        rewritten = rewritten.split("\n\n")[0].strip()
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        logger.info("[graph:rewrite_query] stripped meta-commentary, rewritten=%r", rewritten)
    logger.info("[graph:rewrite_query] OUT rewritten_query=%r", rewritten)
    return {"rewritten_query": rewritten, "iteration": 0}


def _retrieve_node(state: AgentState) -> dict:
    """Node 2: Retrieve context from Milvus + rerank, store chunks, increment iteration."""
    q = (state.get("rewritten_query") or "").strip() or (state.get("original_query") or "").strip()
    it = state.get("iteration") or 0
    logger.info("[graph:retrieve_context] IN  iteration=%d query=%r", it + 1, q)
    chunks = retrieve_context(q)
    sources = [c.get("metadata", {}).get("source", "") for c in chunks]
    logger.info("[graph:retrieve_context] OUT iteration=%d chunks=%d sources=%s", it + 1, len(chunks), sources)
    for i, c in enumerate(chunks):
        text = (c.get("text") or "")
        meta = c.get("metadata") or {}
        logger.info("[graph:retrieve_context] chunk_%d source=%s chunk_id=%s len=%d", i + 1, meta.get("source"), meta.get("chunk_id"), len(text))
        logger.info("[graph:retrieve_context] chunk_%d text=%r", i + 1, text[:400] if len(text) > 400 else text)
    return {"retrieved_chunks": chunks, "iteration": it + 1}


def _normalize_context_for_llm(text: str) -> str:
    """Replace PDF bullet/symbol chars (e.g. \\x7f) with space so the LLM sees readable text."""
    if not text:
        return text
    return text.replace("\x7f", " ").replace("\x00", " ").strip()


def _analyze_context(state: AgentState) -> dict:
    """Node 3: LLM analyses whether the context is sufficient (YES/NO). No keyword override—analysis only."""
    query = state.get("original_query") or ""
    chunks = state.get("retrieved_chunks") or []
    context = "\n\n".join((c.get("text") or "") for c in chunks[:5])
    context_truncated = _normalize_context_for_llm(context[:2000])
    logger.info("[graph:analyze_context] IN  query=%r", query)
    logger.info("[graph:analyze_context] IN  chunks=%d context_len=%d context_truncated_len=%d", len(chunks), len(context), len(context_truncated))
    logger.info("[graph:analyze_context] IN  context_sample=%r", context_truncated[:800])
    prompt = (
        "Does the context mention or explain the topic of the user's question? "
        "If you see the topic name or a clear description (e.g. Burger, Cake, Paternity Leave, Emergency Leave, Ingredients, Instructions), answer YES. "
        "Answer YES when the context mentions the topic at all. Answer NO only if the context is completely irrelevant. "
        "Answer only YES or NO.\n\nUser question: " + query + "\n\nContext:\n" + context_truncated
    )
    logger.info("[graph:analyze_context] prompt_len=%d", len(prompt))
    out = hf_llm(prompt, max_new_tokens=10).strip().upper()
    logger.info("[graph:analyze_context] llm_raw=%r", out)
    needs = "NO" in out or out == "N"
    logger.info("[graph:analyze_context] OUT raw=%r needs_more_context=%s", out, needs)
    return {"needs_more_context": needs}


def _generate_answer(state: AgentState) -> dict:
    """Node 4: LLM generates final answer from retrieved chunks. History-aware."""
    query = state.get("original_query") or ""
    chunks = state.get("retrieved_chunks") or []
    history = state.get("chat_history") or []
    context = "\n\n".join((c.get("text") or "") for c in chunks[:8])
    context_block = _normalize_context_for_llm(context[:4000])
    history_block = _format_history(history)
    logger.info("[graph:generate_answer] IN  query=%r", query)
    logger.info("[graph:generate_answer] IN  chunks=%d context_len=%d context_block_len=%d history_len=%d", len(chunks), len(context), len(context_block), len(history))
    logger.info("[graph:generate_answer] IN  history_block=%r", history_block[:500] if history_block else "")
    logger.info("[graph:generate_answer] IN  context_block_sample=%r", context_block[:1000])
    prompt = f"""
        You are a helpful assistant. Answer using ONLY the provided documents.

        Critical: RELATE YOUR ANSWER TO THE USER'S QUESTION.
        - Address the question directly first, then add relevant details from the documents.
        - Include only information that helps answer what they asked. Do not drift into unrelated topics.
        - Structure your answer around their question: answer what they asked, then briefly support with details from the documents.
        - If they ask about a specific topic, focus on that topic only.

        Writing style:
        - Use a natural, friendly, human tone.
        - Explain the answer smoothly in full sentences.
        - Avoid sounding like a legal or robotic summary.
        - Combine related facts into a clear explanation.
        - Write as if you’re helping a person understand something.

        Accuracy rules:
        - Base your answer ONLY on the provided documents.
        - SEARCH the documents for the user's topic and key terms. If you find a section, heading, or passage about that topic with relevant details (e.g. steps, lists, rules), extract and answer from it. Do NOT say "I couldn't find" when that information is present.
        - The documents may contain bullet points or lists; use whatever structure follows the topic.
        - Do not say you couldn't find information when the documents clearly contain the topic and supporting details.
        - Do not invent or assume missing details.
        - If the answer is truly not present, say: "I couldn't find that information in the documents."
        - Do not mention the words “documents” or “context” in the final answer.
        - Do not summarize unrelated information.

        Conversation:
        {history_block}

        User question:
        {query}

        Documents:
        {context_block}

        Answer (directly address the user's question above):
        """

    logger.info("[graph:generate_answer] prompt_len=%d", len(prompt))
    answer = hf_llm(prompt, max_new_tokens=256).strip()
    logger.info("[graph:generate_answer] OUT answer_len=%d", len(answer))
    logger.info("[graph:generate_answer] OUT answer_full=%r", answer or "")
    return {"answer": answer or "No answer generated."}


def _route_after_analyze(state: AgentState) -> Literal["retrieve_context", "generate_answer"]:
    """If needs more context and iteration < 2, retrieve again; else generate answer."""
    needs = state.get("needs_more_context", False)
    it = state.get("iteration") or 0
    next_node = "retrieve_context" if (needs and it < MAX_ITERATIONS) else "generate_answer"
    logger.info("[graph:route_after_analyze] needs_more=%s iteration=%d max_iter=%d -> %s", needs, it, MAX_ITERATIONS, next_node)
    return next_node


def build_graph():
    """
    Build and compile the agent graph.
    rewrite → retrieve → analyze → (retrieve again if needed, else generate) → END.
    """
    graph = StateGraph(AgentState)

    graph.add_node("rewrite_query", _rewrite_query)
    graph.add_node("retrieve_context", _retrieve_node)
    graph.add_node("analyze_context", _analyze_context)
    graph.add_node("generate_answer", _generate_answer)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("retrieve_context", "analyze_context")
    graph.add_conditional_edges("analyze_context", _route_after_analyze)
    graph.add_edge("generate_answer", END)

    return graph.compile()


def run_agent(question: str, history: list | None = None) -> dict:
    """
    Run the agent synchronously. Returns answer, iterations, chunks_used.
    history: optional list of {"role": "user"|"assistant", "content": str} for context-aware answers.
    """
    if not question or not str(question).strip():
        raise ValueError("question is required")
    q = str(question).strip()
    hist = history if history is not None else []
    logger.info("[run_agent] START question=%r history_len=%d", q, len(hist))
    initial: AgentState = {
        "original_query": q,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "answer": "",
        "needs_more_context": False,
        "iteration": 0,
        "chat_history": hist,
    }
    graph = build_graph()
    final = graph.invoke(initial)
    chunks = final.get("retrieved_chunks") or []
    it = final.get("iteration") or 0
    answer = (final.get("answer") or "").strip()
    logger.info("[run_agent] END iterations=%d chunks_used=%d answer_len=%d", it, len(chunks), len(answer))
    logger.info("[run_agent] END answer=%r", answer)
    return {
        "answer": answer,
        "iterations": it,
        "chunks_used": len(chunks),
    }


def run_agent_stream(question: str, history: list | None = None):
    """
    Run the agent and yield streaming events: rewrite → retrieval → analysis → answer.
    Each yield is {"event": str, "data": str}.
    history: optional list of {"role": "user"|"assistant", "content": str} for context-aware answers.
    """
    if not question or not str(question).strip():
        yield {"event": "error", "data": "question is required"}
        return
    q = str(question).strip()
    hist = history if history is not None else []
    logger.info("[run_agent_stream] START question=%r history_len=%d", q, len(hist))
    initial: AgentState = {
        "original_query": q,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "answer": "",
        "needs_more_context": False,
        "iteration": 0,
        "chat_history": hist,
    }
    graph = build_graph()
    try:
        for event in graph.stream(initial):
            # event: dict mapping node name to state update, e.g. {"rewrite_query": {"rewritten_query": "..."}}
            for node_name, state_update in event.items():
                if node_name == "rewrite_query":
                    yield {"event": "rewrite", "data": state_update.get("rewritten_query", "")}
                elif node_name == "retrieve_context":
                    chunks = state_update.get("retrieved_chunks", [])
                    yield {"event": "retrieval", "data": f"retrieved {len(chunks)} chunks"}
                elif node_name == "analyze_context":
                    needs = state_update.get("needs_more_context", False)
                    yield {"event": "analysis", "data": "context insufficient" if needs else "context sufficient"}
                elif node_name == "generate_answer":
                    yield {"event": "answer", "data": state_update.get("answer", "")}
    except Exception as e:
        logger.exception("[run_agent_stream] Agent stream failed")
        yield {"event": "error", "data": str(e)}
    logger.info("[run_agent_stream] END")


def run_agent_agentic(question: str, history: list | None = None) -> dict:
    """
    Run the agent in agentic (tool-calling) mode. The LLM can call tools:
    search_documents, list_sources, get_chunk, system_stats, current_date, calculator, web_search.
    Requires OPENAI_API_KEY and openai package. Returns answer and list of tools used.
    """
    if not question or not str(question).strip():
        raise ValueError("question is required")
    q = str(question).strip()
    hist = history if history is not None else []
    logger.info("[run_agent_agentic] START question=%r history_len=%d", q, len(hist))

    system_msg = (
        "You are a helpful assistant with access to tools. Use them when needed to answer the user. "
        "Answer based on tool results and conversation history. When you have enough information, "
        "reply with a clear final answer and do not call more tools."
    )
    messages: list[dict] = [{"role": "system", "content": system_msg}]
    for m in hist:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": q})

    tools_used: list[str] = []
    for _ in range(MAX_AGENTIC_ROUNDS):
        content, tool_calls = chat_with_tools(messages, AGENT_TOOLS, max_tokens=512)
        if content and not tool_calls:
            answer = content.strip()
            logger.info("[run_agent_agentic] END answer_len=%d tools_used=%s", len(answer), tools_used)
            return {"answer": answer, "tools_used": tools_used}
        if tool_calls:
            assistant_msg: dict = {"role": "assistant", "content": content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments") or {})}}
                for tc in tool_calls
            ]
            messages.append(assistant_msg)
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments") or {}
                result = execute_tool(name, args)
                tools_used.append(name)
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            continue
        if content:
            answer = content.strip()
            logger.info("[run_agent_agentic] END answer_len=%d tools_used=%s", len(answer), tools_used)
            return {"answer": answer, "tools_used": tools_used}
        break
    answer = "I couldn't complete the request (tool-calling requires OpenAI and OPENAI_API_KEY)."
    logger.info("[run_agent_agentic] END fallback tools_used=%s", tools_used)
    return {"answer": answer, "tools_used": tools_used}
