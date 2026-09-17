"""
rag/rag_index.py
Este modulo realiza:
  - Extracción y chunking de los manuales/normas de protecciones
    (.pdf/.txt/.md).
  - Construcción del índice BM25 (rank_bm25).
  - Todo lo específico de ChromaDB: selección del backend de embeddings,
    construcción de la colección y carga de la colección ya existente
    (usada por rag/retriever.py al consultar).

Ambos índices (BM25 y Chroma) se construyen juntos, en la misma corrida
de build_index(), a partir de la misma lista de chunks -- si
reconstruyes uno sin el otro (p. ej. reindexas solo Chroma con un
chunking distinto) quedan desalineados: rag/retriever.py asume que un
chunk_id significa el mismo texto en ambos índices. Los dos rankings se
combinan por Reciprocal Rank Fusion (RRF) en rag/retriever.py, no aquí

Se ejecuta una vez (o cada vez que cambien los documentos fuente) 
-- no en cada consulta del agente:

    python -m rag.rag_index Manuales --bm25-output manual_bm25.pkl --persist-dir ./chroma_manuales --collection manuales

Acepta .pdf, .txt y .md dentro de la carpeta (recursivo).

POR QUÉ HÍBRIDO (historial de la decisión de diseño):
    BM25: los manuales de protecciones están llenos de
    códigos ANSI/IEEE muy específicos (67P1, 51S1, TRIP, 3PT, 50GF...)
    y BM25 (coincidencia de términos) es más confiable que un embedding
    denso genérico para ESE vocabulario exacto -- un embedding no
    necesariamente distingue bien "67P1" de "67P2".

    ChromaDB (embeddings): dos códigos parecidos
    pueden quedar cerca en el espacio de embeddings aunque semánticamente
    sean elementos de protección distintos.

    Esta versión combina ambos: BM25 aporta precisión en
    vocabulario/códigos exactos, los embeddings aportan generalización
    semántica (sinónimos, paráfrasis, preguntas en lenguaje natural que
    no calzan literalmente con el texto del manual). Se combinan con
    Reciprocal Rank Fusion (RRF) en rag/retriever.py.

IMPORTANTE (backend de embeddings): EMBEDDING_BACKEND (y el modelo
asociado, OPENAI_EMBEDDING_MODEL o ST_EMBEDDING_MODEL) debe ser
EXACTAMENTE el mismo al construir el índice (build_index) y al
consultarlo (rag.retriever, vía load_chroma_collection) 
-- de lo contrario la consulta se embebe en un espacio vectorial distinto al de
los chunks indexados y las distancias dejan de ser comparables.

Requiere:
    pip install rank-bm25 chromadb pypdf
    # backend de embeddings (ver get_embedding_function):
    pip install openai                    # si EMBEDDING_BACKEND=openai (default)
    pip install sentence-transformers     # si EMBEDDING_BACKEND=sentence-transformers
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

# IMPORTANTE: esto tiene que ir ANTES de "import chromadb" 
# En Windows, PyTorch puede colgarse indefinidamente al inicializar 
# su pool de hilos OpenMP/MKL cuando corre dentro de un subproceso 
# SIN consola real (exactamente el caso de un servidor MCP lanzado vía stdio)
# limitar la cantidad de hilos a 1 evita ese cuelgue. os.environ.setdefault
# para no pisar un valor que el usuario ya haya fijado a propósito.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    chunk_id: str


def tokenize(text: str) -> list[str]:
    """Tokenización simple que conserva alfanuméricos pegados (67P1, 3PT,
    TRIP) como un solo token -- separarlos perdería justo la señal más
    útil para este dominio."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


# ---------------------------------------------------------------------------
# Extracción y chunking (independiente del backend de scoring -- ambos
# índices, BM25 y Chroma, se construyen desde estos mismos pedazos de texto)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """
    Divide `text` en fragmentos de hasta `chunk_chars` caracteres,
    intentando cortar en un límite de oración o, si no hay uno cercano,
    de palabra -- nunca a mitad de una palabra ni (cuando es evitable) a
    mitad de una oración. El corte ciego por conteo de caracteres puro
    puede partir una frase, o incluso un renglón de una tabla de ajustes,
    exactamente por la mitad, lo que degrada tanto el matching de BM25
    (mezcla el final de una idea con el principio de otra) como el
    embedding semántico del chunk.

    `chunk_chars` sigue siendo un TAMAÑO OBJETIVO, no una cota estricta:
    se permite extender un poco el corte para llegar al fin de oración
    más cercano, en vez de partirla.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # Busca el límite de oración (". ", "? ", "! ") más cercano
            # hacia atrás desde `end`, sin retroceder más allá de la mitad
            # del chunk (para no generar fragmentos demasiado chicos).
            search_from = max(start + chunk_chars // 2, start)
            boundary = -1
            for sep in (". ", "? ", "! ", "\n"):
                idx = text.rfind(sep, search_from, end)
                if idx > boundary:
                    boundary = idx + len(sep)
            if boundary == -1:
                # Sin límite de oración cercano -- al menos no cortar a
                # mitad de palabra.
                idx = text.rfind(" ", search_from, end)
                boundary = idx + 1 if idx != -1 else end
            end = boundary
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        next_start = max(end - overlap_chars, start + 1)
        # El solapamiento tampoco debería arrancar a mitad de una palabra.
        space_idx = text.find(" ", next_start, end)
        start = space_idx + 1 if space_idx != -1 else next_start
    return chunks


def _extract_pdf(path: Path) -> list[tuple[int | None, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _extract_text_file(path: Path) -> list[tuple[int | None, str]]:
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def _collect_chunks(manuals_dir: Path) -> list[Chunk]:
    """Recorre manuals_dir y produce la lista única de chunks que
    alimenta a AMBOS índices (BM25 y Chroma), en el mismo orden."""
    chunks: list[Chunk] = []

    for path in sorted(manuals_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = _extract_pdf(path)
        elif suffix in (".txt", ".md"):
            pages = _extract_text_file(path)
        else:
            continue

        for page_num, page_text in pages:
            for i, piece in enumerate(_chunk_text(page_text)):
                chunks.append(
                    Chunk(
                        text=piece,
                        source=path.name,
                        # 0 en vez de None: Chroma no acepta None como valor de
                        # metadata, y así el campo "page" queda representado
                        # igual en los dos índices (ver rag/retriever.py, que
                        # traduce 0 -> None de vuelta al responder).
                        page=page_num if page_num is not None else 0,
                        chunk_id=f"{path.name}:p{page_num or 0}:c{i}",
                    )
                )

    if not chunks:
        raise ValueError(f"No se encontraron documentos indexables (.pdf/.txt/.md) en {manuals_dir}")

    return chunks


# ---------------------------------------------------------------------------
# Construcción del índice BM25
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[Chunk], output_path: Path) -> None:
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(output_path, "wb") as f:
        # El orden de `chunks` debe coincidir con el orden usado para
        # construir `bm25` -- get_scores()[i] corresponde a chunks[i].
        # Se persisten juntos precisamente para que nunca se desalineen.
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


# ---------------------------------------------------------------------------
# Backend de embeddings / ChromaDB
# ---------------------------------------------------------------------------

def get_embedding_function():
    backend = os.environ.get("EMBEDDING_BACKEND", "openai").lower()

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_BACKEND=openai (default) requiere OPENAI_API_KEY. "
                "Define la variable de entorno, o usa EMBEDDING_BACKEND=sentence-transformers "
                "para embeddings locales sin API externa."
            )
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model)
    
    if backend == "sentence-transformers":
        model = os.environ.get("ST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)

    raise ValueError(
        f"EMBEDDING_BACKEND desconocido: {backend!r}. Usa 'openai' o 'sentence-transformers'."
    )


# ---------------------------------------------------------------------------
# Construcción (lado de indexado)
# ---------------------------------------------------------------------------

def build_chroma_index(chunks: list[Chunk], persist_dir: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_fn = get_embedding_function()

    # Se recrea la colección desde cero en cada corrida, igual que el
    # pickle de BM25 se sobrescribe entero -- ambos índices deben
    # reflejar exactamente el mismo `chunks`, nunca una mezcla de
    # corridas distintas.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH_SIZE = 100
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "page": c.page} for c in chunks]

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


# ---------------------------------------------------------------------------
# Carga (lado de consulta, rag/retriever.py)
# ---------------------------------------------------------------------------

def load_chroma_collection(persist_dir: str, collection_name: str):
    if not Path(persist_dir).exists():
        raise RuntimeError(
            f"No se encontró el índice Chroma en '{persist_dir}'. Ejecuta primero "
            f"'python -m rag.rag_index <carpeta_de_manuales> --persist-dir {persist_dir} "
            f"--collection {collection_name}' y verifica que CHROMA_PERSIST_DIR/CHROMA_COLLECTION "
            f"apunten ahí."
        )

    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = get_embedding_function()
    try:
        return client.get_collection(name=collection_name, embedding_function=embedding_fn)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo abrir la colección '{collection_name}' en '{persist_dir}'. "
            f"¿Se construyó el índice con 'python -m rag.rag_index'? Detalle: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Construcción de los dos índices juntos
# ---------------------------------------------------------------------------

def build_index(manuals_dir: Path, bm25_output: Path, persist_dir: Path, collection_name: str) -> list[Chunk]:
    chunks = _collect_chunks(manuals_dir)
    build_bm25_index(chunks, bm25_output)
    build_chroma_index(chunks, persist_dir, collection_name)
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Construye el índice híbrido (BM25 + ChromaDB) de manuales para el agente RAG."
    )
    parser.add_argument("manuals_dir", type=Path, help="Carpeta con los manuales (.pdf/.txt/.md)")
    parser.add_argument(
        "--bm25-output", type=Path, default=Path("manual_bm25.pkl"),
        help="Ruta del pickle con el índice BM25 (default: manual_bm25.pkl)",
    )
    parser.add_argument(
        "--persist-dir", type=Path, default=Path("chroma_manuales"),
        help="Carpeta donde ChromaDB persiste la colección (default: ./chroma_manuales)",
    )
    parser.add_argument(
        "--collection", type=str, default="manuales",
        help="Nombre de la colección Chroma (default: 'manuales')",
    )
    args = parser.parse_args()

    chunks = build_index(args.manuals_dir, args.bm25_output, args.persist_dir, args.collection)

    n_docs = len({c.source for c in chunks})
    print(
        f"Índice híbrido construido: {len(chunks)} chunks de {n_docs} documento(s)\n"
        f"  BM25   -> {args.bm25_output}\n"
        f"  Chroma -> {args.persist_dir} (colección '{args.collection}')"
    )


if __name__ == "__main__":
    main()
