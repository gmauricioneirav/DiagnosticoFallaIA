"""
mcp_servers/calculo_server.py

Servidor MCP de cálculo para el agente de diagnóstico de fallas
COMTRADE.

Este es el ÚNICO lugar del sistema donde ocurre aritmética sobre las
señales. El grafo de agentes (graph/workflow.py + agents/*.py) nunca
calcula un valor por su cuenta: solo invoca estas herramientas y razona
sobre lo que devuelven. Ambas herramientas reciben únicamente rutas de
archivo (cfg_path, dat_path) -- nunca arreglos de señales como argumento
-- y devuelven un resumen ya reducido y JSON-serializable.

Herramientas expuestas:
  - extract_fault_features:    envuelve ComtradeFeatureExtractor
                                (tools/comtrade_features.py) -> RMS por
                                ciclo, componentes simétricas de
                                corriente/tensión.
  - extract_protection_status: envuelve SOEExtractor
                                (tools/soe_extractor.py) -> secuencia de
                                actuación de protecciones.

Ejecutar:
    # Transporte stdio (desarrollo local; MultiServerMCPClient lo consume
    # lanzando este script como subproceso -- ver ejemplo de config abajo)
    python mcp_servers/calculo_server.py

    # Transporte HTTP (proceso persistente, para producción o para
    # correr el servidor en otra máquina)
    MCP_TRANSPORT=http MCP_PORT=8001 python mcp_servers/calculo_server.py

Config correspondiente para MCP_SERVERS_CONFIG (usada por
graph.workflow.get_mcp_tools / MultiServerMCPClient) según el transporte
elegido:

    # stdio
    {"calculo": {"transport": "stdio", "command": "python",
                 "args": ["/ruta/a/mcp_servers/calculo_server.py"]}}

    # streamable_http (NOTA: el cliente usa guión bajo "streamable_http";
    # el argumento `transport=` de FastMCP.run(), en cambio, usa guión
    # medio "streamable-http" -- son convenciones de librerías distintas,
    # no lo mismo escrito distinto por error)
    {"calculo": {"transport": "streamable_http", "url": "http://localhost:8001/mcp"}}

Requiere:
    pip install "mcp[cli]" comtrade numpy pandas
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Cuando MultiServerMCPClient lanza este archivo como subproceso
# (`python mcp_servers/calculo_server.py`), Python pone el directorio del
# script (mcp_servers/) como primera entrada de sys.path -- NO la raíz
# del proyecto -- sin importar cuál sea el cwd del proceso padre. Sin
# esto, `from tools.comtrade_features import ...` falla con
# ModuleNotFoundError dentro del subproceso, y langchain-mcp-adapters lo
# reporta de forma confusa como McpError('Connection closed'). Insertar
# la raíz del proyecto (un nivel arriba de mcp_servers/) al principio de
# sys.path lo hace robusto sin importar cómo se invoque el script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from mcp.server.fastmcp import FastMCP

from tools.comtrade_features import ComtradeFeatureExtractor
from tools.soe_extractor import SOEExtractor

mcp = FastMCP(
    "calculo",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8001")),
)


def _dataframe_records(df: pd.DataFrame) -> list[dict]:
    """Convierte un DataFrame a una lista de dicts JSON-serializable,
    reemplazando NaN por None (json no serializa NaN de forma portable)."""
    return df.where(df.notna(), None).to_dict(orient="records")


@mcp.tool()
async def extract_fault_features(cfg_path: str, dat_path: str, include_per_cycle: bool = False) -> dict:
    """Extrae features de señales analógicas de un registro COMTRADE:
    RMS por ciclo y componentes simétricas de corriente y tensión
    (precalculadas por el IED si el archivo las trae, o vía DFT +
    Fortescue como fallback).

    Args:
        cfg_path: ruta al archivo .cfg del registro COMTRADE.
        dat_path: ruta al archivo .dat del registro COMTRADE.
        include_per_cycle: si es True, incluye también la serie completa
            por ciclo (más pesado); por defecto solo se devuelve el
            resumen de la ventana de falla, que es lo que necesita el
            agente de diagnóstico en la mayoría de los casos.

    Returns:
        {
          "summary": {...},              # ComtradeFeatureExtractor.summarize_event
          "per_cycle": [...] | null       # solo si include_per_cycle=True
        }
    """
    def _run():
        extractor = ComtradeFeatureExtractor(cfg_path=cfg_path, dat_path=dat_path).load()
        df = extractor.extract()
        summary = extractor.summarize_event(df)
        return {
            "summary": summary,
            "per_cycle": _dataframe_records(df) if include_per_cycle else None,
        }

    return await asyncio.to_thread(_run)


@mcp.tool()
async def extract_protection_status(
    cfg_path: str, dat_path: str, trip_channel_names: list[str] | None = None
) -> dict:
    """Extrae el estado de actuación de las protecciones (secuencia de
    eventos de los canales digitales) de un registro COMTRADE.

    Args:
        cfg_path: ruta al archivo .cfg del registro COMTRADE.
        dat_path: ruta al archivo .dat del registro COMTRADE.
        trip_channel_names: nombres de canal que representan un disparo
            real para ESTE relé/esquema.
            Sin este dato, la herramienta NO adivina cuál
            activación es "el disparo" -- la primera activación tras el
            trigger suele ser un flag interno (event recorder, arranque
            de un elemento), no el disparo. Si no se conoce aún el
            nombre correcto, se recomienda que el agente lo consulte
            primero vía retrieve_manuals contra el manual del relé.

    Returns:
        Resumen de SOEExtractor.summarize_protection_status(): siempre
        incluye la primera activación tras el trigger (neutral); el
        disparo confirmado ("first_trip_*") solo se llena si
        trip_channel_names matcheó algo.
    """
    def _run():
        soe = SOEExtractor(cfg_path=cfg_path, dat_path=dat_path).load()
        return soe.summarize_protection_status(trip_channel_names=trip_channel_names)

    return await asyncio.to_thread(_run)


if __name__ == "__main__":
    # "http" en la variable de entorno se traduce al literal que espera
    # FastMCP.run() ("streamable-http", con guión medio).
    transport = "streamable-http" if os.environ.get("MCP_TRANSPORT") == "http" else "stdio"
    mcp.run(transport=transport)
