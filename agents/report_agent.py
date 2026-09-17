"""
agents/report_agent.py

Nodo de Salida del grafo ("salida"): produce las gráficas del evento y
redacta el informe técnico final en markdown. Ambas responsabilidades
(generación de figuras vía MCP y redacción del informe vía LLM) se
mantienen en un único nodo del grafo 
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from graph.workflow import TOOL_PLOT_SIGNALS, FaultAnalysisState, call_tool
from prompts.report_prompt import REPORT_SYSTEM_PROMPT


async def generate_figures(cfg_path: str, dat_path: str, fault_window: Optional[dict]) -> dict:
    """Genera las gráficas analógica y digital del evento.

    Invoca la herramienta MCP `plot_signals` (mcp_servers/graficado_server.py).
    No carga señales en Python ni las pasa como argumento -- el servidor MCP
    recibe cfg_path/dat_path y hace su propio ComtradeFeatureExtractor(...).load()
    o SOEExtractor(...).load() para dibujar.
    Args:
        cfg_path: ruta al archivo .cfg del registro COMTRADE.
        dat_path: ruta al archivo .dat del registro COMTRADE.
        fault_window: el sub-dict "fault_window" que devuelve extract_fault_features (con "peak_cycle_time_s"), para que
                      el servidor de graficado resalte la ventana correcta sin tener que redetectarla.
    Returns:
        {"analog": {"format": "png", "data": "<base64>"},
         "digital": {"format": "png", "data": "<base64>"}}
    """
    analog_figure = await call_tool(
        TOOL_PLOT_SIGNALS,
        cfg_path=cfg_path,
        dat_path=dat_path,
        domain="analog",
        fault_window=fault_window,
    )
    digital_figure = await call_tool(
        TOOL_PLOT_SIGNALS,
        cfg_path=cfg_path,
        dat_path=dat_path,
        domain="digital",
        fault_window=fault_window,
    )
    return {"analog": analog_figure, "digital": digital_figure}


async def salida_node(state: FaultAnalysisState) -> dict:
    calc_results = state.get("calc_results") or {}
    fault_window = ((calc_results.get("analog") or {}).get("summary") or {}).get("fault_window")
    figures = await generate_figures(
        cfg_path=state["cfg_path"],
        dat_path=state["dat_path"],
        fault_window=fault_window,
    )
    report_context = {
        "diagnosis_hypothesis": state.get("diagnosis_hypothesis"),
        "confidence": state.get("confidence"),
        "calc_results": state.get("calc_results"),
        "retrieved_docs": state.get("retrieved_docs"),
    }
    report_response = await salida_node.llm.ainvoke(
        [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(report_context, ensure_ascii=False, default=str, indent=2)),
        ]
    )
    report_text = report_response.content
    if state.get("needs_revision"):
        # Se llegó aquí por el tope de seguridad de route_from_supervisor
        # (MAX_REVISION_CYCLES / MAX_SUPERVISOR_STEPS), no porque el
        # Crítico haya dado el visto bueno -- se deja constancia explícita
        # en el propio informe, no solo en el trace interno, para que
        # quien lo lea sepa que este diagnóstico no alcanzó consenso.
        report_text = (
            "> **Nota de trazabilidad**: este diagnóstico se generó tras "
            f"{state.get('revision_count', 0) or 0} ciclo(s) de revisión del Crítico sin "
            "llegar a consenso (última observación: "
            f"\"{state.get('revision_notes') or 'sin detalle'}\"). Se incluye igualmente "
            "como mejor hipótesis disponible, no como diagnóstico validado.\n\n"
        ) + report_text

    return {
        "figures": figures,
        "report_draft": report_text,
        "supervisor_steps": 0 if state.get("next_step") == "salida" else state.get("supervisor_steps", 0) or 0,
    }
