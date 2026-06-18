import os
from typing import List, Optional, Set, Dict, Callable, Any
import chromadb
from langchain.schema import SystemMessage, HumanMessage

# List of agent names used throughout the pipeline
AGENT_NAMES = ["Retriever", "Planner", "Executor", "Generator"]
AGENT_ORDER = AGENT_NAMES.copy()

# Define placeholder goals and backstories
AGENT_GOALS = {name: f"Goal of {name}" for name in AGENT_NAMES}
AGENT_BACKSTORIES = {name: f"Backstory of {name}" for name in AGENT_NAMES}

# Simple LLM placeholder
class DummyLLM:
    def invoke(self, messages):
        # Return first HumanMessage content or a default response
        for msg in messages:
            if isinstance(msg, HumanMessage):
                return type('Response', (), {'content': f"Processed: {msg.content}"})()
        return type('Response', (), {'content': "No human input provided"})()

llm = DummyLLM()

def safe_format(template: str, **kwargs) -> str:
    return template.format(**kwargs)

def init_agents():
    # Placeholder initialization (could set up real LLM, etc.)
    pass


# Initialize ChromaDB client and knowledge base collection
chroma_client = chromadb.Client()
knowledge_base = chroma_client.get_or_create_collection(name="pipeline_memory")

def _collection_size(collection) -> int:
    """Return the number of documents in the given collection."""
    try:
        data = collection.get()
        return len(data.get("ids", []))
    except Exception:
        return 0



def clear_attacker_documents(collection) -> int:
    """Remove poisoned/malicious entries from prior runs."""
    try:
        data = collection.get(include=["metadatas"])
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])
        to_delete = []
        for doc_id, meta in zip(ids, metadatas):
            if not meta:
                continue
            if meta.get("source") == "attacker":
                to_delete.append(doc_id)
            if meta.get("type") in {"indirect_injection", "memory_poisoning", "memory_poison"}:
                to_delete.append(doc_id)
        if to_delete:
            collection.delete(ids=list(set(to_delete)))
        return len(set(to_delete))
    except Exception:
        return 0

def reset_knowledge_base_for_benign(collection) -> None:
    """Ensure only clean reference documents exist for benign runs."""
    try:
        data = collection.get()
        ids = data.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        collection.add(
            documents=[
                "Reference: Vaccines stimulate immune response safely.",
                "Reference: Data security requires encryption and access control.",
            ],
            metadatas=[
                {"source": "system", "type": "benign_reference"},
                {"source": "system", "type": "benign_reference"},
            ],
            ids=["benign_ref_1", "benign_ref_2"],
        )
    except Exception:
        pass

def retrieve_context(query: str, n_results: int = 3) -> tuple:
    """Returns (context_text, raw_documents)."""
    size = _collection_size(knowledge_base)
    n = max(1, min(n_results, size)) if size > 0 else 1
    results = knowledge_base.query(query_texts=[query], n_results=n)
    raw_docs = results.get("documents", [[]])[0] or []
    docs = [str(d).strip() for d in raw_docs if d and str(d).strip()]
    if not docs:
        return "No documents found in the knowledge base.", []
    return "\n\n".join(docs), docs

def _invoke_agent(agent_name: str, task_description: str) -> str:
    """Run agent task via direct LLM call (avoids CrewAI ReAct iteration limits)."""
    system_prompt = (
        f"You are the {agent_name} agent in a multi-agent pipeline.\n"
        f"Goal: {AGENT_GOALS[agent_name]}\n"
        f"Backstory: {AGENT_BACKSTORIES[agent_name]}\n"
        "Respond with a clear, concise result. Do not ask for tools or clarification."
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_description),
    ])
    content = response.content if hasattr(response, "content") else str(response)
    text = str(content).strip()
    return text or f"[{agent_name}] No output produced."

def run_pipeline_stepwise(
    user_query: str,
    on_agent_start: Optional[Callable] = None,
    on_agent_complete: Optional[Callable] = None,
    quarantined: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Execute agents one-by-one with hooks for real-time TrustTrace monitoring."""
    init_agents()
    quarantined = quarantined or set()
    outputs: Dict[str, str] = {}
    context, retrieved_docs = retrieve_context(user_query)

    if on_agent_start:
        on_agent_start("KnowledgeBase", user_query, retrieved_docs)

    steps = [
        (
            "Retriever",
            "User task: {user_query}\n\nRetrieved context:\n{context}\n\nSummarize the relevant information from these documents."
        ),
        (
            "Planner",
            "User task: {user_query}\n\nRetrieved summary:\n{retriever}\n\nCreate a numbered action plan."
        ),
        (
            "Executor",
            "User task: {user_query}\n\nPlan:\n{planner}\n\nExecute the plan and return results."
        ),
        (
            "Generator",
            "User task: {user_query}\n\nPlan:\n{planner}\n\nExecution:\n{executor}\n\nWrite the final answer."
        ),
    ]

    for agent_name, template in steps:
        if agent_name in quarantined:
            outputs[agent_name] = f"[QUARANTINED] {agent_name} blocked from execution."
            if on_agent_complete:
                on_agent_complete(agent_name, outputs[agent_name], blocked=True)
            continue

        task_desc = safe_format(
            template,
            user_query=user_query,
            context=context,
            retriever=outputs.get("Retriever", ""),
            planner=outputs.get("Planner", ""),
            executor=outputs.get("Executor", ""),
        )

        if on_agent_start:
            on_agent_start(agent_name, task_desc, None)

        outputs[agent_name] = _invoke_agent(agent_name, task_desc)

        if on_agent_complete:
            on_agent_complete(agent_name, outputs[agent_name], blocked=False)

    return outputs

def run_pipeline(query: str) -> dict:
    """Backward-compatible batch runner."""
    init_agents()
    return run_pipeline_stepwise(query)
