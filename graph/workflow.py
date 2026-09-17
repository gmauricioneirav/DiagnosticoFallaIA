"""
graph/workflow.py

Motor del grafo de agentes (LangGraph) para el diagnóstico de fallas
eléctricas a partir de registros COMTRADE. 

Implementa la arquitectura Supervisor + Blackboard: un nodo 
Supervisor (agents/coordinator.py) decide en cada iteración, 
vía salida estructurada del LLM, cuál especialista ejecutar a 
continuación -- no hay un orden fijo entre los agentes: Ingesta, 
Features, RAG, Diagnóstico, Crítico y Salida.

Este módulo es el "motor": define el State compartido (blackboard), la
infraestructura de acceso a herramientas MCP (carga, caché, invocación),
el decorador de trazabilidad y build_graph(), que ensambla los nodos
definidos en agents/*.py en un StateGraph compilado. Los propios nodos
NO viven aquí -- viven en agents/, cada uno en su archivo.

Principio central: ningún nodo hace aritmética por su cuenta. 
Los nodos que necesitan cálculos (Features,Salida) invocan 
herramientas MCP (cálculo, graficado, retrieval) cargadas vía 
langchain-mcp-adapters. El único cómputo que corre en
Python puro es el parseo del archivo COMTRADE (Ingesta), que es
extracción de estructura, no un cálculo de diagnóstico.

Contrato de State (debe calzar con lo que espera app.py):
    diagnosis_hypothesis: str
    confidence: float (0-1)
    report_draft: str (markdown)
    figures: {"analog": {...}, "digital": {...}}   # format/data
    messages: historial de mensajes (para el chat)

NOTA IMPORTANTE sobre imports: este módulo NO importa nada de agents/ a
nivel de módulo -- solo dentro de build_graph(), como import local. Los
módulos de agents/ sí importan de aquí (call_tool, TOOL_*, get_react_agent,
FaultAnalysisState) a nivel de módulo. Si graph/workflow.py importara
agents/* arriba del archivo, se formaría un ciclo de imports (agents
necesita graph.workflow, graph.workflow necesitaría agents) que rompería
la carga del paquete. Mantener los imports de agents/* como locales,
solo dentro de build_graph(), evita el ciclo sin perder la separación de
archivos pedida.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import operator
import os
import sys
import time
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# El nombre efectivo de este logger es "graph.workflow" (== __name__,
# dado que este archivo vive en el paquete graph/). 
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nombres de herramientas MCP 
# ---------------------------------------------------------------------------
TOOL_EXTRACT_FEATURES = "extract_fault_features"
TOOL_EXTRACT_PROTECTION_STATUS = "extract_protection_status"
TOOL_PLOT_SIGNALS = "plot_signals"
TOOL_RETRIEVE_MANUALS = "retrieve_manuals"
TOOL_BUILD_REPORT = "build_report_document"


# ---------------------------------------------------------------------------
# 1. State compartido (blackboard)
# ---------------------------------------------------------------------------
class FaultAnalysisState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    trace: Annotated[list[dict], operator.add]  # Registro de auditoría: una entrada por cada paso del grafo que se ejecutó, con su duración. 
        
    cfg_path: Optional[str]
    dat_path: Optional[str]

    raw_metadata: Optional[dict]
    features: Optional[dict]
    calc_results: Optional[dict]
    retrieved_docs: Optional[list[dict]]
    rag_attempted: Optional[bool]               # distingue "RAG ya corrió" de "RAG corrió y no encontró nada"
    rag_retried_for_revision: Optional[bool]    # True mientras 'rag' ya se re-invocó para EL CICLO DE REVISIÓN ACTUAL (ver critico_agent.py/needs_revision) -- se resetea a False en cada diagnostico_node, que es lo que marca el inicio de un ciclo nuevo. Sin esto, el Supervisor ve el mismo needs_revision=True/revision_notes sin cambios en cada vuelta (rag_node no los toca) y puede repetir 'rag' indefinidamente hasta chocar con MAX_SUPERVISOR_STEPS -- ver la regla correspondiente en prompts/coordinator_prompt.py.

    diagnosis_hypothesis: Optional[str]
    confidence: Optional[float]
    needs_revision: Optional[bool]
    reviewed_by_critico: Optional[bool]         # se resetea a False en cada diagnostico_node; True solo tras pasar por critico_node
    revision_count: Optional[int]               # ciclos crítico->diagnóstico consecutivos con needs_revision=True; usado como tope de seguridad (ver route_from_supervisor)
    supervisor_steps: Optional[int]             # cuántas veces corrió el Supervisor en el turno actual; tope de seguridad GENERAL (ver route_from_supervisor), se resetea a 0 al empezar un turno nuevo (ver agents/coordinator.py: supervisor_node) o al llegar a 'salida' de forma genuina
    human_messages_seen: Optional[int]          # cuántos mensajes humanos había la última vez que corrió el Supervisor -- usado para detectar "empezó un turno de chat nuevo" y darle presupuesto fresco de supervisor_steps (ver agents/coordinator.py: supervisor_node)
    revision_notes: Optional[str]

    figures: Optional[dict]
    report_draft: Optional[str]

    next_step: Optional[str]


# ---------------------------------------------------------------------------
# 2. Carga de herramientas MCP
# ---------------------------------------------------------------------------
_MCP_TOOLS_CACHE: Optional[list[BaseTool]] = None

def _build_servers_config() -> dict:
    """
    Arma la config de MultiServerMCPClient a partir de variables de entorno:
         MCP_CALCULO_SCRIPT     (p. ej. mcp_servers/calculo_server.py)
         MCP_GRAFICADO_SCRIPT   (p. ej. mcp_servers/graficado_server.py)
         MCP_RETRIEVAL_SCRIPT   (p. ej. mcp_servers/retrieval_server.py)
         BM25_INDEX_PATH        (default: manual_bm25.pkl) -- pickle BM25 construido con `python -m rag.loader`
         CHROMA_PERSIST_DIR     (default: chroma_manuales) -- carpeta del índice ChromaDB construido con `python -m rag.loader`
         CHROMA_COLLECTION      (default: manuales)
         EMBEDDING_BACKEND      (default: openai) -- debe coincidir con el usado al construir el índice; ver rag/vector_store.py
       Los tres servidores se lanzan como subproceso local (transport
       "stdio") ejecutando `sys.executable <script>` -- es decir, el
       MISMO intérprete de Python que está corriendo este proceso (el de
       Streamlit).
    """
    calculo_script = os.environ.get("MCP_CALCULO_SCRIPT")
    graficado_script = os.environ.get("MCP_GRAFICADO_SCRIPT")
    retrieval_script = os.environ.get("MCP_RETRIEVAL_SCRIPT")
    bm25_index_path = os.environ.get("BM25_INDEX_PATH")
    chroma_persist_dir = os.environ.get("CHROMA_PERSIST_DIR")
    chroma_collection = os.environ.get("CHROMA_COLLECTION")
    embedding_backend = os.environ.get("EMBEDDING_BACKEND")

    if not any([calculo_script, graficado_script, retrieval_script]):
        return {}

    servers: dict = {}
    if calculo_script:
        servers["calculo"] = {"transport": "stdio", "command": sys.executable, "args": [calculo_script]}
    if graficado_script:
        servers["graficado"] = {"transport": "stdio", "command": sys.executable, "args": [graficado_script]}
    if retrieval_script:
        retrieval_env = dict(os.environ)  # hereda todo (OPENAI_API_KEY, proxy, etc.)
        retrieval_env.update(
            {
                "BM25_INDEX_PATH": bm25_index_path or "manual_bm25.pkl",
                "CHROMA_PERSIST_DIR": chroma_persist_dir or "chroma_manuales",
                "CHROMA_COLLECTION": chroma_collection or "manuales",
                "EMBEDDING_BACKEND": embedding_backend or "openai",
            }
        )
        servers["retrieval"] = {"transport": "stdio", "command": sys.executable, "args": [retrieval_script], "env": retrieval_env,}
    return servers


async def get_mcp_tools() -> list[BaseTool]:
    """Carga las herramientas de todos los servidores MCP configurados.
       Se cachean en memoria del proceso tras la primera llamada. 
    """
    global _MCP_TOOLS_CACHE
    if _MCP_TOOLS_CACHE is not None:
        return _MCP_TOOLS_CACHE

    servers = _build_servers_config()
    if not servers:
        _MCP_TOOLS_CACHE = []
        return _MCP_TOOLS_CACHE

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(servers)
    try:
        _MCP_TOOLS_CACHE = await client.get_tools()
    except Exception as exc:
        raise RuntimeError(
            "No se pudo iniciar uno o más servidores MCP "
            f"({', '.join(servers.keys())}). Causas típicas: el script no "
            "existe en la ruta configurada (MCP_CALCULO_SCRIPT / "
            "MCP_GRAFICADO_SCRIPT / MCP_RETRIEVAL_SCRIPT), el servidor "
            "lanza una excepción al arrancar (prueba ejecutarlo manualmente, "
            "p. ej. `python mcp_servers/calculo_server.py`, para ver el "
            "error real), o falta una dependencia de requirements.txt en "
            f"este entorno. Detalle original: {exc!r}"
        ) from exc
    return _MCP_TOOLS_CACHE


async def get_tool(name: str) -> Optional[BaseTool]:
    tools = await get_mcp_tools()
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _parse_mcp_result(raw: Any) -> Any:
    """
    langchain-mcp-adapters devuelve tool.ainvoke() como una LISTA de
    bloques de contenido MCP ({'type': 'text', 'text': '<json>', 'id': ...}),
    incluso cuando la herramienta MCP devolvió un único objeto -- en ese
    caso la lista trae un solo bloque. Si la herramienta devolvió una
    lista de varios objetos (como retrieve_manuals), cada elemento de esa
    lista llega como un bloque de contenido SEPARADO, no como un único
    bloque con un array JSON adentro.

    Reconstruye la forma original:
      - 0 bloques -> None
      - 1 bloque  -> el objeto parseado (dict, normalmente)
      - N bloques -> lista de objetos parseados
    """
    if not isinstance(raw, list):
        return raw  # por si una versión futura de la librería cambia el contrato

    parsed = []
    for block in raw:
        text = block.get("text") if isinstance(block, dict) else None
        if text is None:
            parsed.append(block)
            continue
        try:
            parsed.append(json.loads(text))
        except json.JSONDecodeError:
            parsed.append(text)

    if not parsed:
        return None
    if len(parsed) == 1:
        return parsed[0]
    return parsed


async def call_tool(name: str, **kwargs) -> Any:
    """Invoca una herramienta MCP por nombre y devuelve su resultado ya
    parseado (ver _parse_mcp_result). Lanza RuntimeError explícito si la
    herramienta no está disponible -- nunca cae a un cálculo local de
    respaldo.

    Cada invocación queda registrada en el log del proceso con su
    duración (`logger`) -- útil para auditar, independientemente de la
    UI, cuánto tiempo se fue en cada herramienta MCP en particular
    
    Tiene un timeout (MCP_TOOL_TIMEOUT_S, default 60s) alrededor de
    tool.ainvoke(): sin esto, si el subproceso de la herramienta se
    queda colgado -- p. ej. una llamada de red a un backend de
    embeddings que nunca responde porque falta una credencial en el
    entorno del subproceso, o un proxy corporativo no heredado -- el
    grafo entero se queda esperando para siempre sin ningún error ni
    indicio en el log"""
    tool = await get_tool(name)
    if tool is None:
        raise RuntimeError(
            f"La herramienta MCP '{name}' no está disponible. Verifica "
            f"MCP_SERVERS_CONFIG y que el servidor correspondiente esté activo."
        )
    timeout_s = float(os.environ.get("MCP_TOOL_TIMEOUT_S", "60"))
    t0 = time.perf_counter()
    logger.info("[mcp] iniciando herramienta: %s", name)
    try:
        raw = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        elapsed = time.perf_counter() - t0
        logger.error("[mcp] %s excedió el timeout de %.0fs (colgada %.1fs)", name, timeout_s, elapsed)
        raise RuntimeError(
            f"La herramienta MCP '{name}' no respondió en {timeout_s:.0f}s (timeout). "
            f"tiempo. Detalle: {exc!r}"
        ) from exc
    logger.info("[mcp] %s completada en %.3fs", name, time.perf_counter() - t0)
    return _parse_mcp_result(raw)


_REACT_AGENT_CACHE: dict[str, Any] = {}


def _safe_for_log(value: Any, max_len: int = 300) -> Any:
    """
    Reduce un valor del State a algo compacto y serializable para el log
    de trazabilidad -- NO busca ser una copia exacta reconstruible, sino
    legible: trunca strings largos (p.ej. el PNG en base64 de una figura,
    o un fragmento largo de manual recuperado por RAG) y convierte
    mensajes de LangChain a su forma más simple {"type", "content"}.
    """
    if isinstance(value, str):
        return value if len(value) <= max_len else f"{value[:max_len]}... [{len(value)} chars]"
    if isinstance(value, dict):
        return {k: _safe_for_log(v, max_len) for k, v in value.items()}
    if isinstance(value, list):
        head = [_safe_for_log(v, max_len) for v in value[:10]]
        return head + [f"... (+{len(value) - 10} más)"] if len(value) > 10 else head
    if hasattr(value, "content") and hasattr(value, "type"):  # AIMessage/HumanMessage/...
        return {"type": value.type, "content": _safe_for_log(value.content, max_len)}
    return value


def _traced(step_name: str):
    """
    Envuelve un nodo del grafo (sync o async) para:
      1. Medir cuánto tarda en ejecutarse.
      2. Dejar constancia de ese paso en el log del proceso.
      3. Añadir una entrada a `state["trace"]` con {"step", "duration_s"}
    """
    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(state, *args, **kwargs):
                t0 = time.perf_counter()
                result = dict(await fn(state, *args, **kwargs) or {})
                duration = time.perf_counter() - t0
                extra = result.pop("_trace_extra", None) or {}
                written = {k: _safe_for_log(v) for k, v in result.items() if k != "trace"}
                logger.info(
                    "[grafo] %s completado en %.3fs -- escribió: %s",
                    step_name, duration, list(written.keys()),
                )
                result["trace"] = [
                    {"step": step_name, "duration_s": round(duration, 3), "written": written, **extra}
                ]
                return result
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(state, *args, **kwargs):
            t0 = time.perf_counter()
            result = dict(fn(state, *args, **kwargs) or {})
            duration = time.perf_counter() - t0
            extra = result.pop("_trace_extra", None) or {}
            written = {k: _safe_for_log(v) for k, v in result.items() if k != "trace"}
            logger.info(
                "[grafo] %s completado en %.3fs -- escribió: %s",
                step_name, duration, list(written.keys()),
            )
            result["trace"] = [
                {"step": step_name, "duration_s": round(duration, 3), "written": written, **extra}
            ]
            return result
        return sync_wrapper
    return decorator


async def get_react_agent(llm, tool_names: list[str], cache_key: str):
    """Construye (una sola vez por cache_key) un agente ReAct de LangGraph
        con el subconjunto de herramientas MCP indicado.
    """
    if cache_key in _REACT_AGENT_CACHE:
        return _REACT_AGENT_CACHE[cache_key]

    from langgraph.prebuilt import create_react_agent

    all_tools = await get_mcp_tools()
    selected = [t for t in all_tools if t.name in tool_names]
    agent = create_react_agent(llm, selected)
    _REACT_AGENT_CACHE[cache_key] = agent
    return agent


# ---------------------------------------------------------------------------
# 3. Construcción del grafo
# ---------------------------------------------------------------------------

_CHECKPOINTER = MemorySaver()  # Guarda en memoria RAM
_COMPILED_GRAPH: Optional[Any] = None


def build_graph(llm):
    """Construye (una sola vez por proceso) el StateGraph completo.

    Se cachea junto con un checkpointer compartido para que, aunque
    app.py llame a build_graph(llm) en cada turno, el historial del
    caso (thread_id) se preserve entre el análisis inicial y el chat.

    Los nodos se importan aquí (import local, no arriba del archivo) a
    propósito 
    """
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH

    from agents.coordinator import build_supervisor_node, route_from_supervisor
    from agents.comtrade_agent import ingesta_node
    from agents.feature_agent import features_node
    from agents.rag_agent import rag_node
    from agents.diagnosis_agent import diagnostico_node
    from agents.critico_agent import build_critico_node
    from agents.report_agent import salida_node

    # Los nodos que necesitan el LLM pero no lo reciben como argumento
    # (por la firma fija que exige LangGraph) lo guardan como atributo
    # de función -- patrón simple para no reescribir toda la firma de nodos.
    diagnostico_node.llm = llm
    salida_node.llm = llm

    graph = StateGraph(FaultAnalysisState)

    graph.add_node("supervisor", _traced("supervisor")(build_supervisor_node(llm)))
    graph.add_node("ingesta", _traced("ingesta")(ingesta_node))
    graph.add_node("features", _traced("features")(features_node))
    graph.add_node("rag", _traced("rag")(rag_node))
    graph.add_node("diagnostico", _traced("diagnostico")(diagnostico_node))
    graph.add_node("critico", _traced("critico")(build_critico_node(llm)))
    graph.add_node("salida", _traced("salida")(salida_node))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "ingesta": "ingesta",
            "features": "features",
            "rag": "rag",
            "diagnostico": "diagnostico",
            "critico": "critico",
            "salida": "salida",
            END: END,
        },
    )

    # Todo especialista retorna al Supervisor para la siguiente decisión.
    for node_name in ("ingesta", "features", "rag", "diagnostico", "critico", "salida"):
        graph.add_edge(node_name, "supervisor")

    _COMPILED_GRAPH = graph.compile(checkpointer=_CHECKPOINTER)
    return _COMPILED_GRAPH
