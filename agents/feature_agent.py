"""
agents/feature_agent.py

Nodo de Features: calcula magnitudes de las señales (RMS por ciclo,
componentes simétricas) y el estado de actuación de las protecciones,
invocando herramientas MCP de cálculo -- nunca calcula nada en Python
Dos llamadas MCP, ninguna con señales como argumento: 
el servidor de cálculo (mcp_servers/calculo_server.py) hace su propio load() de
tools/comtrade_features.py y tools/soe_extractor.py respectivamente. 
El primero da evidencia de la forma de onda (RMS, secuencias); 
el segundo da evidencia de qué protección operó y cuándo 
-- ambos necesarios para que Diagnóstico dictamine el tipo de falla
"""

from __future__ import annotations

from graph.workflow import (
    TOOL_EXTRACT_FEATURES,
    TOOL_EXTRACT_PROTECTION_STATUS,
    FaultAnalysisState,
    call_tool,
)


async def features_node(state: FaultAnalysisState) -> dict:
    analog_result = await call_tool(
        TOOL_EXTRACT_FEATURES,
        cfg_path=state["cfg_path"],
        dat_path=state["dat_path"],
    )
    protection_status = await call_tool(
        TOOL_EXTRACT_PROTECTION_STATUS,
        cfg_path=state["cfg_path"],
        dat_path=state["dat_path"],
    )

    calc_results = {"analog": analog_result, "protection": protection_status}
    features = {
        "analog_summary": analog_result.get("summary"),
        "protection_summary": protection_status,
    }
    return {"calc_results": calc_results, "features": features}
