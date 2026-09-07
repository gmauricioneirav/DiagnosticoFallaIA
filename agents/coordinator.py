"""
agents/coordinator.py

Nodo Supervisor/Coordinador del grafo: en cada iteración decide, vía
salida estructurada del LLM, cuál especialista ejecutar a continuación
(ingesta, features, rag, diagnostico, critico, salida, o terminar).
No calcula ni diagnostica nada por su cuenta -- solo enruta, a partir
de un resumen compacto del State actual (blackboard).

También expone route_from_supervisor(), la función de enrutamiento
condicional que usa graph/workflow.py, con los topes de seguridad
DETERMINÍSTICOS que evitan que el grafo quede en bucle si el LLM no
sigue bien las reglas del prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from pydantic import BaseModel, Field

from graph.workflow import FaultAnalysisState
from prompts.coordinator_prompt import SUPERVISOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilidades de contexto
# ---------------------------------------------------------------------------

def _last_human_text(state: FaultAnalysisState) -> Optional[str]:
    for msg in reversed(state.get("messages", [])):
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            return msg.content if hasattr(msg, "content") else msg.get("content")
    return None


def _count_human_messages(state: FaultAnalysisState) -> int:
    """Cuenta solo mensajes humanos/de usuario -- los nodos especialistas
    (p. ej. diagnostico_node) agregan mensajes del asistente en cada
    vuelta, así que contar TODOS los mensajes haría que esto creciera en
    cada paso dentro del MISMO turno, no solo cuando llega una pregunta
    nueva. Ver supervisor_node: esto es la señal para detectar "empezó un
    turno nuevo" y darle presupuesto fresco de supervisor_steps."""
    count = 0
    for msg in state.get("messages", []) or []:
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            count += 1
    return count


def _last_message_pending_answer(state: FaultAnalysisState) -> bool:
    """True si el último mensaje de la conversación es del usuario --
    es decir, todavía nadie (ni diagnostico_node) le respondió. Sin esto,
    una pregunta de chat que llega después de que el caso ya cerró
    ('end') se quedaría sin respuesta: la cascada vería
    tiene_diagnostico/revisado_por_critico/tiene_figuras/tiene_informe
    todos en verdadero (de la corrida anterior) y volvería a 'end' de
    inmediato.
    """
    messages = state.get("messages", [])
    if not messages:
        return False
    last = messages[-1]
    role = getattr(last, "type", None) or (last.get("role") if isinstance(last, dict) else None)
    return role in ("human", "user")


def _summarize_state_for_supervisor(state: FaultAnalysisState) -> str:
    summary = {
        "tiene_raw_metadata": bool(state.get("raw_metadata")),
        "tiene_features": bool(state.get("features")),
        # bool(retrieved_docs) sería incorrecto: una consulta legítima sin
        # coincidencias devuelve [], que es falsy en Python -- eso haría
        # que el Supervisor crea que RAG nunca corrió y vuelva a
        # invocarlo indefinidamente. rag_attempted distingue "ya corrió"
        # de "corrió y no encontró nada".
        "tiene_retrieved_docs": bool(state.get("rag_attempted")),
        "tiene_diagnostico": bool(state.get("diagnosis_hypothesis")),
        "confianza_actual": state.get("confidence"),
        "revisado_por_critico": bool(state.get("reviewed_by_critico")),
        "needs_revision": state.get("needs_revision", False),
        "revision_notes": state.get("revision_notes"),
        "rag_reintentado_para_esta_revision": bool(state.get("rag_retried_for_revision")),
        "revision_count": state.get("revision_count", 0) or 0,
        "tiene_figuras": bool(state.get("figures")),
        "tiene_informe": bool(state.get("report_draft")),
        "ultima_pregunta_usuario": _last_human_text(state),
        "pregunta_sin_responder": _last_message_pending_answer(state),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Nodo Supervisor (enrutamiento condicional, no determinístico)
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    next: Literal["ingesta", "features", "rag", "diagnostico", "critico", "salida", "end"] = Field(
        description="Siguiente especialista a ejecutar, o 'end' si el caso ya está resuelto "
        "y no hay una pregunta pendiente del usuario."
    )
    rationale: str = Field(description="Razonamiento breve (1-2 frases) de la decisión.")


def build_supervisor_node(llm):
    structured_llm = llm.with_structured_output(RouteDecision)

    def supervisor_node(state: FaultAnalysisState) -> dict:
        context = _summarize_state_for_supervisor(state)
        decision = structured_llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=f"Estado actual del caso:\n{context}"),
            ]
        )

        # Detecta si llegó una pregunta NUEVA del usuario desde la última
        # vez que corrió este nodo, contando solo mensajes humanos (ver
        # _count_human_messages). Cada invocación de invoke_graph() desde
        # app.py para una pregunta de chat agrega EXACTAMENTE un mensaje
        # humano nuevo -- si el conteo creció desde la última corrida de
        # este nodo, es la PRIMERA vez que el Supervisor ve esta
        # pregunta y merece presupuesto fresco de MAX_SUPERVISOR_STEPS,
        # sin importar que el turno anterior haya agotado el suyo -- 
        # incluso si terminó forzado por el freno de seguridad (ver
        # route_from_supervisor). Sin esto, supervisor_steps queda
        # "contaminado" para siempre (persiste en el checkpoint entre
        # turnos, vía el mismo thread_id) y CUALQUIER pregunta de chat
        # posterior a un cierre forzado terminaría el grafo de inmediato
        # sin ejecutar ningún nodo -- sin responder nunca al usuario.
        current_human_count = _count_human_messages(state)
        last_seen_human_count = state.get("human_messages_seen", 0) or 0
        is_new_turn = current_human_count > last_seen_human_count

        prior_supervisor_steps = 0 if is_new_turn else (state.get("supervisor_steps", 0) or 0)

        return {
            "next_step": decision.next,
            # Contador general de pasos del Supervisor DENTRO del turno
            # actual -- tope de seguridad determinístico en
            # route_from_supervisor, independiente de CUÁL sea el patrón
            # de bucle (a diferencia de revision_count, que solo cubre el
            # ciclo diagnostico<->critico). Se reinicia a 0 cuando
            # is_new_turn es verdadero (ver arriba), y también cuando
            # agents/report_agent.py detecta una salida NO forzada.
            "supervisor_steps": prior_supervisor_steps + 1,
            "human_messages_seen": current_human_count,
            # Se fusiona en la entrada de trace de este nodo (ver _traced) y
            # no se persiste como clave propia del State -- deja rastro de
            # POR QUÉ el Supervisor decidió ir a `decision.next`, útil para
            # reconstruir el paso a paso completo del análisis.
            "_trace_extra": {"decision": decision.next, "rationale": decision.rationale},
        }

    return supervisor_node


# Topes de seguridad DETERMINÍSTICOS -- no dependen de que el LLM del
# Supervisor interprete bien el prompt. Dos niveles, del más específico
# al más general:
#
# 1. MAX_REVISION_CYCLES: ciclos crítico->diagnóstico consecutivos con
#    needs_revision=True. Cubre el caso "Crítico y Diagnóstico no logran
#    converger" -- se dispara temprano y de forma dirigida.
#
# 2. MAX_SUPERVISOR_STEPS: tope GENERAL de cuántas veces corrió el
#    Supervisor en esta corrida, sin importar el patrón de bucle. Existe
#    porque en la práctica un solo "ciclo de revisión" puede consumir más
#    de 2 pasos del grafo (p. ej. el Supervisor pidiendo 'rag' dos veces
#    seguidas antes de volver a 'diagnostico'), así que MAX_REVISION_CYCLES
#    por sí solo puede no alcanzar a dispararse antes de que LangGraph
#    llegue a su recursion_limit por defecto (25) y aborte con
#    GraphRecursionError -- justo lo que se observó en producción. Este
#    tope es la red de respaldo para CUALQUIER bucle no anticipado, no
#    solo el de crítico/diagnóstico.
#
# supervisor_steps se resetea a 0 en agents/report_agent.py (salida_node)
# -- cada intento de diagnóstico (incluida una re-apertura del caso por
# una pregunta nueva del usuario que exige recalcular) arranca con
# presupuesto fresco.
MAX_REVISION_CYCLES = 2
MAX_SUPERVISOR_STEPS = 10


def route_from_supervisor(state: FaultAnalysisState) -> str:
    mapping = {
        "ingesta": "ingesta",
        "features": "features",
        "rag": "rag",
        "diagnostico": "diagnostico",
        "critico": "critico",
        "salida": "salida",
        "end": END,
    }
    next_step = state.get("next_step")
    revision_count = state.get("revision_count", 0) or 0
    supervisor_steps = state.get("supervisor_steps", 0) or 0

    # Si supervisor_steps YA supera el tope (no solo lo alcanza) es porque
    # una vuelta anterior de este mismo bucle ya forzó 'salida' una vez
    # (ver agents/report_agent.py: salida_node NO resetea supervisor_steps
    # a 0 cuando fue una salida forzada, precisamente para que esta rama
    # se dispare aquí) y el Supervisor, tras eso, TODAVÍA no logró decidir
    # 'end' por su cuenta -- es la señal inequívoca de un bucle real (p.
    # ej. el LLM insistiendo en 'rag' una y otra vez). En ese caso hay que
    # terminar el grafo de verdad (END), no volver a forzar 'salida' --
    # de lo contrario 'salida' resetearía el contador otra vez y el
    # Supervisor tendría presupuesto fresco para volver a atascarse,
    # repitiendo el ciclo indefinidamente hasta chocar con el
    # recursion_limit duro de LangGraph (GraphRecursionError sin
    # oportunidad de cerrar limpio).
    if next_step not in ("salida", "end") and supervisor_steps > MAX_SUPERVISOR_STEPS:
        logger.warning(
            "[grafo] el tope general de pasos ya se había forzado antes y el Supervisor "
            "sigue sin decidir 'end' (%d pasos) -- terminando el grafo (END) para evitar "
            "un bucle infinito supervisor<->salida",
            supervisor_steps,
        )
        return END

    if next_step not in ("salida", "end") and supervisor_steps >= MAX_SUPERVISOR_STEPS:
        logger.warning(
            "[grafo] tope general de pasos del Supervisor alcanzado (%d) -- forzando 'salida'",
            supervisor_steps,
        )
        return "salida"

    if next_step in ("diagnostico", "rag") and revision_count >= MAX_REVISION_CYCLES:
        logger.warning(
            "[grafo] tope de ciclos de revisión alcanzado (%d) -- forzando 'salida'",
            revision_count,
        )
        return "salida"

    # Freno determinístico para la regla "tras diagnostico, SIEMPRE
    # critico, sin excepción salvo pregunta_sin_responder" -- no depender
    # de que el LLM no confunda esto con la regla, distinta, de
    # 'ultima_pregunta_usuario' (que solo autoriza saltar pasos
    # redundantes como ingesta/features/rag, nunca critico). Sin este
    # freno se observó en producción 'diagnostico' corriendo 3 veces
    # seguidas sin pasar por 'critico' en el medio -- el LLM tomó la
    # existencia de 'ultima_pregunta_usuario' (que casi siempre existe)
    # como licencia para volver directo a 'diagnostico', ignorando que
    # 'revisado_por_critico' seguía en falso.
    diagnosis_exists = bool(state.get("diagnosis_hypothesis"))
    reviewed_by_critico = bool(state.get("reviewed_by_critico"))
    pending_answer = _last_message_pending_answer(state)
    if diagnosis_exists and not reviewed_by_critico and not pending_answer and next_step != "critico":
        logger.warning(
            "[grafo] hay un diagnóstico sin revisar por el crítico y no hay pregunta "
            "pendiente del usuario, pero el Supervisor decidió '%s' -- forzando 'critico'",
            next_step,
        )
        return "critico"

    # Freno determinístico específico para el patrón "rag repetido":
    # si el LLM del Supervisor decide 'rag' pero YA se reintentó rag para
    # el ciclo de revisión actual (ver rag_agent.py/diagnosis_agent.py),
    # no depender de que haya seguido bien la regla del prompt -- forzar
    # 'diagnostico' directamente. needs_revision/revision_notes no
    # cambian entre una llamada a 'rag' y la siguiente (solo
    # diagnostico/critico los tocan), así que sin este freno el LLM ve
    # básicamente el mismo estado en cada vuelta y puede repetir 'rag'
    # hasta agotar MAX_SUPERVISOR_STEPS -- justo el bucle observado en
    # producción (7 llamadas seguidas a retrieve_manuals en un turno).
    if next_step == "rag" and state.get("rag_attempted") and state.get("rag_retried_for_revision"):
        logger.warning(
            "[grafo] el Supervisor pidió 'rag' de nuevo pero ya se había reintentado para "
            "este ciclo de revisión -- forzando 'diagnostico' en su lugar"
        )
        return "diagnostico"

    return mapping.get(next_step, "ingesta")
