"""
rag/retriever.py

Lógica de retrieval (RAG) en tiempo de consulta sobre el índice HÍBRIDO
ya construido (ver rag/rag_index.py: BM25 + ChromaDB sobre el mismo
conjunto de chunks). No hace ninguna llamada a un LLM ni decide nada por
su cuenta: solo recupera texto ya existente en los documentos indexados,
combinando dos rankings ya calculados (BM25 y similitud de embeddings)
por Reciprocal Rank Fusion (RRF).

Este módulo es puro (sin FastMCP) para poder testearlo o reutilizarlo
sin levantar un servidor -- mcp_servers/retrieval_server.py es apenas un
wrapper delgado que expone retrieve_manuals() de aquí como herramienta
MCP.

POR QUÉ HÍBRIDO + RRF (ver también el docstring de rag/rag_index.py):
    BM25 es más confiable para el vocabulario técnico exacto de estos
    manuales (códigos ANSI/IEEE como 67P1, 51S1, TRIP, 3PT, 50GF...),
    donde un embedding denso genérico puede no distinguir bien "67P1" de
    "67P2". Los embeddings, en cambio, generalizan mejor ante preguntas
    en lenguaje natural que no calzan literalmente con el texto del
    manual (sinónimos, paráfrasis). RRF combina ambos rankings SIN
    normalizar sus escalas de score (que no son comparables entre sí:
    BM25 no está acotado, la similitud coseno sí) -- solo usa la
    POSICIÓN de cada documento en cada ranking:

        RRF(d) = sum_r  1 / (k + rank_r(d))

    sumado sobre cada ranking r en el que aparece d (si un chunk solo
    aparece en un ranking, solo aporta ese término). k=60 es el valor
    estándar de la literatura (Cormack et al., 2009) y de por sí ya
    amortigua la diferencia entre estar en la posición 1 vs. la 2,
    haciendo la fusión robusta a que un solo método "gane" por poco.

Antes de usar, hay que construir el índice híbrido:
    python -m rag.rag_index /ruta/a/manuales \
        --bm25-output manual_bm25.pkl --persist-dir ./chroma_manuales --collection manuales

Requiere:
    pip install rank-bm25 chromadb
    # más el backend de embeddings usado (openai o sentence-transformers,
    # ver rag/rag_index.py)
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Optional

from rag.rag_index import Chunk, load_chroma_collection, tokenize

# Constante estándar de RRF (Cormack, Clarke & Buettcher, 2009). Config-
# urable por si se quiere experimentar, pero 60 es el valor de referencia.
RRF_K = int(os.environ.get("RRF_K", "60"))

# Cuántos candidatos se piden a CADA ranker antes de fusionar -- debe ser
# mayor que top_k para que la fusión tenga margen real de recombinar
# (si solo se pidiera top_k a cada uno, un chunk relevante que quede en
# la posición top_k+1 de BM25 pero #1 en embeddings nunca entraría).
_CANDIDATE_MULTIPLIER = int(os.environ.get("RRF_CANDIDATE_MULTIPLIER", "4"))
_MIN_CANDIDATES = int(os.environ.get("RRF_MIN_CANDIDATES", "20"))

_BM25_CACHE: Optional[dict] = None
_CHROMA_COLLECTION_CACHE = None


# ---------------------------------------------------------------------------
# Carga perezosa de ambos índices (cacheados en memoria del proceso)
# ---------------------------------------------------------------------------

def _load_bm25_index() -> dict:
    global _BM25_CACHE
    if _BM25_CACHE is not None:
        return _BM25_CACHE

    index_path = os.environ.get("BM25_INDEX_PATH", "manual_bm25.pkl")
    if not Path(index_path).exists():
        raise RuntimeError(
            f"No se encontró el índice BM25 en '{index_path}'. Ejecuta primero "
            f"'python -m rag.rag_index <carpeta_de_manuales> --bm25-output {index_path} ...' "
            f"y verifica que BM25_INDEX_PATH apunte ahí."
        )
    with open(index_path, "rb") as f:
        try:
            _BM25_CACHE = pickle.load(f)
        except (AttributeError, ModuleNotFoundError) as exc:
            _BM25_CACHE = _load_legacy_main_pickle(index_path, exc)
    return _BM25_CACHE


def _load_legacy_main_pickle(index_path: str, original_exc: Exception) -> dict:
    """
    Compatibilidad con pickles generados corriendo el script
    directamente (`python rag_index.py ...` o `python rag\\rag_index.py
    ...`) en vez de con `python -m rag.rag_index ...`. En ese caso
    `Chunk` quedó serializado bajo el módulo "__main__" DE AQUEL
    proceso, que no existe aquí.

    En vez de obligar a reconstruir el índice (potencialmente caro si
    hay muchos documentos y se usa un backend de embeddings pagado), se
    inyecta temporalmente `Chunk` como atributo del "__main__" de ESTE
    proceso -- que es exactamente donde pickle lo va a buscar -- y se
    reintenta. Es solo un parche de lectura: el archivo en disco no se
    modifica, así que sigue siendo buena idea reconstruirlo con `-m`
    quien quiera evitar este parche a futuro.
    """
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        main_module.Chunk = Chunk

    try:
        with open(index_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo deserializar el índice BM25 en '{index_path}' ni siquiera con "
            f"el parche de compatibilidad ({exc}). Reconstruye el índice con:\n"
            f"    python -m rag.rag_index <carpeta_de_manuales> --bm25-output {index_path} ...\n"
            "(no 'python rag_index.py ...' ni 'python rag\\rag_index.py ...')."
        ) from original_exc


def _load_chroma_collection():
    global _CHROMA_COLLECTION_CACHE
    if _CHROMA_COLLECTION_CACHE is not None:
        return _CHROMA_COLLECTION_CACHE

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "chroma_manuales")
    collection_name = os.environ.get("CHROMA_COLLECTION", "manuales")
    _CHROMA_COLLECTION_CACHE = load_chroma_collection(persist_dir, collection_name)
    return _CHROMA_COLLECTION_CACHE


# ---------------------------------------------------------------------------
# Los dos rankings individuales
# ---------------------------------------------------------------------------

def _bm25_ranking(query: str, candidate_k: int, sources: Optional[list[str]]) -> list[dict]:
    """Retorna una lista ordenada (rank 1 = mejor) de hasta candidate_k
    resultados BM25, cada uno con text/source/page/chunk_id/raw_score.
    Nunca incluye resultados con score<=0 (sin coincidencia real de
    término -- igual que en la versión BM25 pura)."""
    index = _load_bm25_index()
    bm25 = index["bm25"]
    chunks = index["chunks"]

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked_idx:
        if scores[i] <= 0:
            break  # el resto tampoco tendrá coincidencias reales de término
        chunk = chunks[i]
        if sources and chunk.source not in sources:
            continue
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page or None,
                "raw_score": float(scores[i]),
            }
        )
        if len(results) >= candidate_k:
            break
    return results


def _chroma_ranking(query: str, candidate_k: int, sources: Optional[list[str]]) -> list[dict]:
    """Retorna una lista ordenada (rank 1 = mejor) de hasta candidate_k
    resultados por similitud de embeddings, con la misma forma que
    _bm25_ranking (text/source/page/chunk_id/raw_score)."""
    collection = _load_chroma_collection()
    where = {"source": {"$in": sources}} if sources else None

    result = collection.query(
        query_texts=[query],
        n_results=candidate_k,
        where=where,
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    results = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        meta = meta or {}
        results.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": meta.get("source"),
                "page": meta.get("page") or None,
                # similitud coseno (1.0 = idéntico); solo informativo, RRF
                # no usa este valor, solo la posición en esta lista.
                "raw_score": float(1.0 - distance),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    rankings: dict[str, list[dict]], top_k: int, rrf_k: int = RRF_K
) -> list[dict]:
    """
    Combina N rankings (cada uno una lista ya ordenada de mejor a peor)
    en uno solo, sin necesitar que sus scores sean comparables entre sí.

    rankings: {"bm25": [...], "vector": [...]}, cada lista con dicts que
    incluyen "chunk_id".

    Retorna una lista fusionada, ordenada por score RRF descendente,
    recortada a top_k, con cada item anotado con su rank/score de origen
    en cada método (para trazabilidad -- no para volver a puntuar).
    """
    fused: dict[str, dict] = {}

    for method, ranked_list in rankings.items():
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            entry = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "text": item["text"],
                    "source": item["source"],
                    "page": item["page"],
                    "rrf_score": 0.0,
                    "matched_by": {},
                },
            )
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["matched_by"][method] = {"rank": rank, "raw_score": item["raw_score"]}

    ordered = sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)
    return ordered[:top_k]


# ---------------------------------------------------------------------------
# Punto de entrada usado por mcp_servers/retrieval_server.py
# ---------------------------------------------------------------------------

def retrieve_manuals(query: str, top_k: int = 5, sources: Optional[list[str]] = None) -> list[dict]:
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
        vector) y su rank/score en cada uno -- útil para depurar si un
        resultado vino solo de coincidencia léxica, solo de similitud
        semántica, o de ambos (más confiable). Lista vacía si ningún
        método encontró nada -- nunca inventa un resultado.
    """
    query_text = (query or "").strip()
    if not query_text:
        return []

    candidate_k = max(_MIN_CANDIDATES, top_k * _CANDIDATE_MULTIPLIER)

    bm25_results = _bm25_ranking(query_text, candidate_k, sources)
    vector_results = _chroma_ranking(query_text, candidate_k, sources)

    fused = _reciprocal_rank_fusion(
        {"bm25": bm25_results, "vector": vector_results}, top_k=top_k
    )

    return [
        {
            "text": item["text"],
            "source": item["source"],
            "page": item["page"],
            "score": item["rrf_score"],
            "matched_by": item["matched_by"],
        }
        for item in fused
    ]
