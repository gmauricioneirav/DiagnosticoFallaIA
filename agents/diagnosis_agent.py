"""
agents/diagnosis_agent.py

Nodo de Diagnóstico: agente ReAct de LangGraph que genera o actualiza la
hipótesis de falla y su confianza, combinando features y contexto
normativo. Puede invocar herramientas MCP de cálculo o de retrieval si
necesita un dato adicional que no esté ya en el contexto -- nunca
inventa cifras.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.coordinator import _last_human_text
from graph.workflow import (
    TOOL_EXTRACT_FEATURES,
    TOOL_EXTRACT_PROTECTION_STATUS,
    TOOL_RETRIEVE_MANUALS,
    FaultAnalysisState,
    get_react_agent,
)
from prompts.diagnosis_prompt import DIAGNOSTICO_SYSTEM_PROMPT


class DiagnosisResult(BaseModel):
    hypothesis: str = Field(description="Diagnóstico de la falla en lenguaje natural.")
    confidence: float = Field(description="Nivel de confianza entre 0 y 1.", ge=0.0, le=1.0)


async def diagnostico_node(state: FaultAnalysisState) -> dict:
    agent = await get_react_agent(
        llm=diagnostico_node.llm,
        tool_names=[TOOL_EXTRACT_FEATURES, TOOL_EXTRACT_PROTECTION_STATUS, TOOL_RETRIEVE_MANUALS],
        cache_key="diagnostico",
    )

    context = {
        "features": state.get("features"),
        "retrieved_docs": state.get("retrieved_docs"),
        "revision_notes": state.get("revision_notes"),
    }
    human_content = (
        f"Contexto disponible:\n{json.dumps(context, ensure_ascii=False, default=str, indent=2)}\n\n"
        f"Pregunta o tarea actual: {_last_human_text(state) or 'Genera el diagnóstico inicial del evento.'}"
    )

    agent_result = await agent.ainvoke(
        {"messages": [SystemMessage(content=DIAGNOSTICO_SYSTEM_PROMPT), HumanMessage(content=human_content)]}
    )
    final_message = agent_result["messages"][-1]

    structured = await diagnostico_node.llm.with_structured_output(DiagnosisResult).ainvoke(
        [
            SystemMessage(content="Extrae del siguiente análisis la hipótesis final y la confianza (0-1)."),
            HumanMessage(content=final_message.content),
        ]
    )

    return {
        "diagnosis_hypothesis": structured.hypothesis,
        "confidence": structured.confidence,
        # Ambas banderas se resetean aquí, no solo reviewed_by_critico:
        # needs_revision describía una inconsistencia de la hipótesis
        # ANTERIOR. Si no se limpia también aquí, queda en True para
        # siempre tras el primer rechazo del Crítico -- la regla del
        # Supervisor ("needs_revision=True -> vuelve a diagnostico/rag,
        # nunca a critico") entonces reenvía a este nodo indefinidamente,
        # porque solo critico_node puede volver a poner needs_revision en
        # False, y la propia regla se lo impide mientras siga en True.
        # Este fue justo el bucle observado en producción.
        "needs_revision": False,
        "reviewed_by_critico": False,  # toda hipótesis nueva/actualizada exige pasar por critico de nuevo
        # Igual razón que needs_revision arriba: esta hipótesis es NUEVA,
        # así que si el próximo critico vuelve a pedir más contexto
        # normativo, es un pedido legítimo y distinto -- merece su propio
        # reintento de 'rag', no heredar el "ya reintenté" de la
        # hipótesis anterior. Ver rag_agent.py y coordinator_prompt.py.
        "rag_retried_for_revision": False,
        "messages": [AIMessage(content=final_message.content)],
    }
