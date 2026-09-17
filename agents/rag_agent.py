"""
agents/rag_agent.py

Nodo RAG: consulta los manuales/normas de protecciones relevantes,
invocando la herramienta MCP `retrieve_manuals` 
(servida por mcp_servers/retrieval_server.py, que a su vez usa rag/retriever.py).
No interpreta el contenido recuperado.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.coordinator import _last_human_text
from graph.workflow import TOOL_RETRIEVE_MANUALS, FaultAnalysisState, call_tool

# Umbral por debajo del cual una componente de secuencia (cero/negativa)
# se considera "presente" 
_SEQ_PRESENCE_RATIO = 0.1


def _fmt(value: Any, unit: str = "", decimals: int = 1) -> Optional[str]:
    """Formatea un número para el query; devuelve None si no hay dato
    (evita ensuciar el query con 'None' o 'nan')."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return f"{value:.{decimals}f}{unit}"


def _describe_event(state: FaultAnalysisState) -> list[str]:
    """Arma fragmentos de texto en lenguaje natural con la información
    específica del evento disponible en el State, para que el query del
    RAG apunte a los criterios de protección relevantes para ESTE caso
    (tipo de falla, fases involucradas, protección que operó, relé involucrado).
    """
    parts: list[str] = []

    raw_metadata = state.get("raw_metadata") or {}
    if raw_metadata.get("device_id"):
        parts.append(f"relé {raw_metadata['device_id']}")
    if raw_metadata.get("station_name"):
        parts.append(f"subestación {raw_metadata['station_name']}")

    features = state.get("features") or {}
    analog_summary = features.get("analog_summary") or {}
    fault_window = analog_summary.get("fault_window") or {}

    if fault_window:
        current_rms = fault_window.get("current_rms") or {}
        currents = ", ".join(
            f"I{ph}={v}"
            for ph, v in (
                ("a", _fmt(current_rms.get("a"), " A")),
                ("b", _fmt(current_rms.get("b"), " A")),
                ("c", _fmt(current_rms.get("c"), " A")),
            )
            if v is not None
        )
        if currents:
            parts.append(f"corrientes de falla {currents}")

        voltage_rms = fault_window.get("voltage_rms") or {}
        voltages = ", ".join(
            f"V{ph}={v}"
            for ph, v in (
                ("a", _fmt(voltage_rms.get("a"), " V")),
                ("b", _fmt(voltage_rms.get("b"), " V")),
                ("c", _fmt(voltage_rms.get("c"), " V")),
            )
            if v is not None
        )
        if voltages:
            parts.append(f"tensiones durante la falla {voltages}")
        # Secuencia negativa/cero presentes en magnitud apreciable frente
        # a la positiva => indicio de asimetría (fase-fase / fase-tierra)
        # que vale la pena nombrar explícitamente en el query.
        current_seq = fault_window.get("current_sequence") or {}
        pos_mag = (current_seq.get("positive") or {}).get("magnitude")
        neg_mag = (current_seq.get("negative") or {}).get("magnitude")
        zero_mag = (current_seq.get("zero") or {}).get("magnitude")
        if pos_mag:
            if neg_mag and neg_mag / pos_mag > _SEQ_PRESENCE_RATIO:
                parts.append("presencia significativa de secuencia negativa de corriente (falla asimétrica)")
            if zero_mag and zero_mag / pos_mag > _SEQ_PRESENCE_RATIO:
                parts.append("presencia significativa de secuencia cero de corriente (posible falla a tierra)")

    protection_summary = features.get("protection_summary") or {}
    activations = protection_summary.get("activations_after_trigger") or []
    operated_channels: list[str] = []
    for act in activations:
        name = act.get("channel") if isinstance(act, dict) else None
        if name and name not in operated_channels:
            operated_channels.append(name)

    if operated_channels:
        parts.append(f"protecciones que operaron: {', '.join(operated_channels)}")
    else:
        trip_channel = protection_summary.get("first_trip_channel") or protection_summary.get(
            "first_activation_channel"
        )
        if trip_channel:
            parts.append(f"protección que operó: canal {trip_channel}")
    first_trip_channel = protection_summary.get("first_trip_channel")
    if first_trip_channel:
        parts.append(f"disparo confirmado: canal {first_trip_channel}")

    return parts


async def rag_node(state: FaultAnalysisState) -> dict:
    hypothesis = state.get("diagnosis_hypothesis")
    user_question = _last_human_text(state)
    event_parts = _describe_event(state)

    query_parts = [p for p in [hypothesis, *event_parts, user_question] if p]
    query = " | ".join(query_parts) or "criterios generales de protección para el evento registrado"

    docs = await call_tool(TOOL_RETRIEVE_MANUALS, query=query, top_k=5)
    # call_tool() no puede distinguir "la herramienta devolvió None" de "devolvió una lista vacía" 
    # (0 bloques de contenido MCP en ambos casos) -- por eso None se normaliza aquí a lista vacía, 
    # no a [None], que ensuciaría retrieved_docs con un resultado falso.
    if docs is None:
        docs = []
    elif not isinstance(docs, list):
        docs = [docs]
    return {
        "retrieved_docs": docs,
        "rag_attempted": True,
        "rag_retried_for_revision": True,
    }
