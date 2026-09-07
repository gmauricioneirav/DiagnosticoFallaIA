"""
mcp_servers/retrieval_server.py

Servidor MCP de retrieval (RAG) para el agente de diagnóstico de
fallas. Es un wrapper delgado: toda la lógica de búsqueda (BM25 +
ChromaDB + Reciprocal Rank Fusion) vive en rag/retriever.py -- este
archivo solo la expone como herramienta MCP.

No hace ninguna llamada a un LLM ni decide nada por su cuenta: solo
recupera texto ya existente en los documentos indexados (ver
rag/rag_index.py: BM25 + ChromaDB sobre el mismo conjunto de chunks). La
interpretación de esos fragmentos (qué implican para el diagnóstico)
queda del lado del agente de Diagnóstico (agents/diagnosis_agent.py) --
este servidor nunca resume, parafrasea ni añade nada al contenido
recuperado.

Antes de usar, hay que construir el índice híbrido:
    python -m rag.rag_index /ruta/a/manuales \
        --bm25-output manual_bm25.pkl --persist-dir ./chroma_manuales --collection manuales

Ejecutar:
    BM25_INDEX_PATH=manual_bm25.pkl CHROMA_PERSIST_DIR=./chroma_manuales \
        CHROMA_COLLECTION=manuales python mcp_servers/retrieval_server.py
    MCP_TRANSPORT=http MCP_PORT=8003 python mcp_servers/retrieval_server.py

IMPORTANTE: EMBEDDING_BACKEND (y el modelo asociado, OPENAI_EMBEDDING_MODEL
o ST_EMBEDDING_MODEL) debe ser EXACTAMENTE el mismo que se usó al
construir el índice con `python -m rag.rag_index`.

Requiere:
    pip install "mcp[cli]" rank-bm25 chromadb
    # más el backend de embeddings usado (openai o sentence-transformers,
    # ver rag/rag_index.py)

BLINDAJE DE STDOUT (leer antes de tocar este archivo):
    Con transporte "stdio", MCP usa el stdout (fd 1) del proceso como el
    canal EXCLUSIVO del protocolo JSON-RPC -- cada línea que se escribe
    ahí tiene que ser exactamente un mensaje bien formado. Si CUALQUIER
    código (una librería como sentence-transformers/huggingface_hub con
    una barra de progreso o un warning impreso con print() en vez de
    logging, chromadb, o incluso un print() propio de debug) escribe
    algo más en stdout, ese mensaje queda corrupto: el subproceso puede
    terminar de calcular perfectamente bien, pero el cliente MCP nunca
    recibe algo que pueda parsear y se queda esperando PARA SIEMPRE --
    indistinguible de un cuelgue real, salvo que no hay ningún error en
    ningún lado.

    Por eso, antes de cualquier import que pueda imprimir algo (incluido
    "from rag.retriever import ..."), este archivo duplica el fd 1 real
    en uno "limpio" aparte y redirige el fd 1 del proceso a stderr para
    TODO lo demás. El servidor, al arrancar en modo stdio, usa
    explícitamente ese duplicado limpio para el protocolo en vez de
    dejar que FastMCP tome sys.stdout tal cual (que a partir de acá
    apunta a stderr) -- ver _run_stdio_blindado() más abajo.
"""

import os
import sys

# --- Blindaje de stdout: DEBE ir antes de cualquier otro import ---
# Guarda un duplicado "limpio" del stdout real (para el protocolo MCP)
# y redirige el fd 1 del proceso hacia stderr para todo lo demás. Ver
# la nota larga en el docstring de arriba.
_CLEAN_STDOUT_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import asyncio
import io
from pathlib import Path

import anyio

# Ver la nota equivalente en mcp_servers/calculo_server.py: sin esto,
# `from rag.retriever import ...` falla dentro del subproceso porque
# sys.path[0] queda apuntando a mcp_servers/, no a la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

from rag.retriever import retrieve_manuals as _retrieve_manuals_core

mcp = FastMCP(
    "retrieval",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8003")),
)


@mcp.tool()
async def retrieve_manuals(query: str, top_k: int = 5, sources: list[str] | None = None) -> list[dict]:
    """Busca los fragmentos más relevantes de los manuales/normas
    indexados, combinando BM25 (coincidencia de términos, fuerte en
    códigos ANSI/IEEE exactos) y embeddings densos vía ChromaDB (fuerte
    en similitud semántica/paráfrasis) por Reciprocal Rank Fusion.

    Args:
        query: consulta en lenguaje natural o términos técnicos (p. ej.
            nombres de elementos de protección, tipo de falla sospechado,
            nombre de un canal digital cuyo significado se necesita).
        top_k: cuántos fragmentos devolver como máximo (por defecto 5).
        sources: si se indica, restringe la búsqueda a estos nombres de
            archivo exactos (p. ej. ["SEL-451-Manual.pdf"]). Útil cuando
            ya se sabe qué manual de fabricante aplica.

    Returns:
        Lista de {"text", "source", "page", "score", "matched_by"},
        ordenada por score RRF descendente. "matched_by" indica, para
        trazabilidad, en qué ranking(s) apareció el fragmento (bm25 y/o
        vector) y su rank/score en cada uno. Lista vacía si ningún
        método encontró nada -- nunca inventa un resultado.
    """
    return await asyncio.to_thread(_retrieve_manuals_core, query, top_k, sources)


async def _run_stdio_blindado() -> None:
    """
    Equivalente a mcp.run(transport="stdio") / FastMCP.run_stdio_async(),
    pero pasándole explícitamente el descriptor de stdout LIMPIO
    (guardado antes de cualquier import ruidoso, ver el blindaje al
    inicio del archivo) en vez de dejar que tome sys.stdout tal cual
    (que a esta altura apunta a stderr). Así, pase lo que pase durante
    la ejecución de una tool (una barra de progreso, un warning, un
    print() de alguna dependencia), esa salida cae en stderr y nunca
    corrompe el canal del protocolo MCP.

    Se accede a mcp._mcp_server porque la API pública de FastMCP no
    expone una forma de inyectar streams personalizados para stdio; es
    exactamente lo que hace FastMCP.run_stdio_async() internamente (ver
    mcp/server/fastmcp/server.py), solo que aquí se le pasa `stdout`
    explícito a stdio_server() en vez del default.
    """
    clean_stdout = anyio.wrap_file(
        io.TextIOWrapper(os.fdopen(_CLEAN_STDOUT_FD, "wb", closefd=False), encoding="utf-8")
    )
    async with stdio_server(stdout=clean_stdout) as (read_stream, write_stream):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )


if __name__ == "__main__":
    if os.environ.get("MCP_TRANSPORT") == "http":
        # El transporte HTTP no usa stdout como canal -- no necesita el
        # blindaje de arriba, mcp.run() normal está bien acá.
        mcp.run(transport="streamable-http")
    else:
        anyio.run(_run_stdio_blindado)
