"""
agents/rag_agent.py

Nodo RAG: consulta los manuales/normas de protecciones relevantes,
invocando la herramienta MCP `retrieve_manuals` (servida por
mcp_servers/retrieval_server.py, que a su vez usa rag/retriever.py).
No interpreta el contenido recuperado -- eso es trabajo del agente de
Diagnóstico.
"""

from __future__ import annotations

from agents.coordinator import _last_human_text
from graph.workflow import TOOL_RETRIEVE_MANUALS, FaultAnalysisState, call_tool


async def rag_node(state: FaultAnalysisState) -> dict:
    hypothesis = state.get("diagnosis_hypothesis")
    user_question = _last_human_text(state)

    # TODO: mejorar la construcción de la consulta (p.ej. incluir tipo de
    # falla sospechado, fase afectada, fabricante del relé).
    query_parts = [p for p in [hypothesis, user_question] if p]
    query = " | ".join(query_parts) or "criterios generales de protección para el evento registrado"

    docs = await call_tool(TOOL_RETRIEVE_MANUALS, query=query, top_k=5)
    # call_tool() no puede distinguir "la herramienta devolvió None" de
    # "devolvió una lista vacía" (0 bloques de contenido MCP en ambos
    # casos) -- por eso None se normaliza aquí a lista vacía, no a
    # [None], que ensuciaría retrieved_docs con un resultado falso.
    if docs is None:
        docs = []
    elif not isinstance(docs, list):
        docs = [docs]
    return {
        "retrieved_docs": docs,
        "rag_attempted": True,
        # Marca "ya reintenté rag para el ciclo de revisión actual" --
        # diagnostico_node lo resetea a False en cada corrida (ver ese
        # archivo), así que este flag naturalmente vuelve a estar
        # disponible para un ÚNICO reintento por cada nueva hipótesis.
        # Sin esto, el Supervisor podía repetir 'rag' indefinidamente
        # mientras needs_revision/revision_notes no cambiaran (ver
        # prompts/coordinator_prompt.py).
        "rag_retried_for_revision": True,
    }
