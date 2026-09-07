"""
agents/critico_agent.py

Nodo Crítico: revisa si el diagnóstico es consistente con los
resultados de cálculo y con el contexto normativo recuperado, antes de
que el caso pueda cerrarse. No invoca herramientas MCP -- solo razona
sobre lo que ya está en el State.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.workflow import FaultAnalysisState
from prompts.critico_prompt import CRITICO_SYSTEM_PROMPT


class CriticResult(BaseModel):
    consistent: bool = Field(description="True si el diagnóstico es consistente con cálculos y normativa.")
    notes: str = Field(description="Si no es consistente, qué debería revisarse y por qué.")


def build_critico_node(llm):
    structured_llm = llm.with_structured_output(CriticResult)

    def critico_node(state: FaultAnalysisState) -> dict:
        payload = {
            "diagnosis_hypothesis": state.get("diagnosis_hypothesis"),
            "confidence": state.get("confidence"),
            "calc_results": state.get("calc_results"),
            "retrieved_docs": state.get("retrieved_docs"),
        }
        result = structured_llm.invoke(
            [
                SystemMessage(content=CRITICO_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str, indent=2)),
            ]
        )
        prev_count = state.get("revision_count", 0) or 0
        return {
            "needs_revision": not result.consistent,
            "revision_notes": result.notes,
            "reviewed_by_critico": True,
            # Se incrementa solo si volvió a ser inconsistente; se resetea
            # a 0 apenas el Crítico da el visto bueno. route_from_supervisor
            # usa esto como tope de seguridad determinístico (ver ahí) --
            # independiente de si el LLM interpreta bien needs_revision.
            "revision_count": prev_count + 1 if not result.consistent else 0,
        }

    return critico_node
