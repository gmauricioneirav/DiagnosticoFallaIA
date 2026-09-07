# FaultDiagnosisAI

Sistema multiagente (LangGraph + servidores MCP) para diagnóstico de
fallas eléctricas a partir de registros COMTRADE.

Este README documenta la reorganización de carpetas hecha sobre el
proyecto original (todo en un puñado de archivos sueltos: `app.py`,
`supervisor_agent.py`, `comtrade_features.py`, `channel_identifier.py`,
`soe_extractor.py`, `mcp_calculo_server.py`, `mcp_graficado_server.py`,
`mcp_retrieval_server.py`, `build_manual_index.py`, `manual_chunking.py`)
hacia la estructura actual. La lógica y el comportamiento **no
cambiaron** -- esto es una reorganización de archivos, no una reescritura.

## Estructura

```
FaultDiagnosisAI/
│
├── app.py                    # Interfaz Streamlit (presentación pura)
├── requirements.txt
│
├── agents/                    # Un archivo por nodo del grafo
│   ├── coordinator.py          # Supervisor: decide next_step (antes en supervisor_agent.py)
│   ├── comtrade_agent.py       # Nodo "ingesta": parseo puro del archivo COMTRADE
│   ├── signal_agent.py         # Genera las gráficas (analógica/digital) vía MCP
│   ├── feature_agent.py        # Nodo "features": RMS, componentes simétricas, protecciones
│   ├── rag_agent.py            # Nodo "rag": consulta manuales normativos vía MCP
│   ├── diagnosis_agent.py      # Nodo "diagnostico": agente ReAct
│   ├── critico_agent.py        # Nodo "critico": validación de consistencia
│   └── report_agent.py         # Nodo "salida": orquesta signal_agent + redacta el informe
│
├── graph/
│   └── workflow.py             # State compartido, infraestructura MCP, tracing, build_graph()
│
├── mcp_servers/                # Los 3 servidores MCP (wrappers delgados sobre tools/ y rag/)
│   ├── calculo_server.py
│   ├── graficado_server.py
│   └── retrieval_server.py
│
├── tools/                      # Extracción/cálculo determinístico (sin LLM)
│   ├── comtrade_features.py    # RMS, DFT, Fortescue, ComtradeFeatureExtractor
│   ├── channel_identifier.py   # Identificación de canales por fabricante
│   └── soe_extractor.py        # Secuencia de eventos de canales digitales
│
├── rag/                        # Índice híbrido BM25 + ChromaDB
│   ├── rag_index.py             # Chunking/extracción de manuales + construcción de BM25 y ChromaDB + CLI de indexado
│   └── retriever.py             # BM25 + Chroma + Reciprocal Rank Fusion (lógica pura, sin FastMCP)
│
├── prompts/                    # Un archivo por system prompt
│   ├── coordinator_prompt.py
│   ├── diagnosis_prompt.py
│   ├── critico_prompt.py
│   └── report_prompt.py
│
├── reports/                     # Salida de informes/logs generados en tiempo de ejecución
└── sample/                      # Coloca aquí tus .cfg/.dat de ejemplo
```

## Decisiones de mapeo (y por qué)

- **`mcp_servers/` en vez de `mcp/`**: si la carpeta se llamara `mcp/`
  chocaría con el paquete `pip install mcp` (el SDK oficial de Model
  Context Protocol) que usan los 3 servidores vía
  `from mcp.server.fastmcp import FastMCP`. Con el nombre `mcp/`, Python
  resolvería la carpeta del proyecto en vez de la librería instalada y
  los tres servidores dejarían de arrancar.

- **`agents/rag_agent.py` y `agents/critico_agent.py`**: el grafo
  original tiene nodos `rag` y `critico` que no aparecían en la lista de
  archivos pedida (`coordinator`, `comtrade_agent`, `signal_agent`,
  `feature_agent`, `diagnosis_agent`, `report_agent`). Se agregaron como
  archivos adicionales para no perder esa separación de responsabilidades.

- **`tools/comtrade_features.py`, `channel_identifier.py`,
  `soe_extractor.py` casi intactos** (en vez de partidos en
  `rms.py`/`dft.py`/`symmetrical.py`/`comtrade_reader.py`): se mantienen
  como archivos propios completos dentro de `tools/` para minimizar el
  riesgo de romper el cálculo de componentes simétricas / RMS al partir
  una clase que hoy comparte estado interno (`channel_map`, `rec`).

- **`agents/signal_agent.py` + `agents/report_agent.py`**: el nodo
  `salida` original generaba figuras Y el informe en un solo paso. Se
  separó la generación de figuras a `signal_agent.py`
  (`generate_figures()`), pero `report_agent.py` sigue siendo el único
  nodo `"salida"` del grafo (invoca `generate_figures` y luego redacta el
  informe) -- así el prompt del Supervisor y `RouteDecision` (que tratan
  `"salida"` como un solo paso) no tuvieron que cambiar.

- **`graph/workflow.py` importa `agents/*` de forma LOCAL** (dentro de
  `build_graph()`, no arriba del archivo): los módulos de `agents/`
  necesitan `call_tool`, `TOOL_*`, `get_react_agent` y
  `FaultAnalysisState` de `graph.workflow`. Si `graph/workflow.py`
  importara `agents/*` a nivel de módulo también, se formaría un ciclo.

- **`rag/loader.py` y `rag/vector_store.py` se consolidaron en
  `rag/rag_index.py`**: la construcción del índice BM25 y de la
  colección ChromaDB corren ambas dentro de `build_index()`, en la
  misma pasada sobre la misma lista de chunks -- reconstruir un índice
  sin el otro los desalinea (`rag/retriever.py` asume que un mismo
  `chunk_id` significa el mismo texto en los dos). Por eso quedaron
  como un único módulo en vez de dos.

## Cómo correr

1. `pip install -r requirements.txt`
2. Crear un archivo `.env` (este repo no trae `.env.example`) con al
   menos `OPENAI_API_KEY` (u `LLM_PROVIDER=ollama` para modelo local),
   y las variables que use tu configuración de servidores MCP -- ver
   `_build_servers_config()` en `graph/workflow.py` para la lista
   completa (`MCP_CALCULO_SCRIPT`, `MCP_GRAFICADO_SCRIPT`,
   `MCP_RETRIEVAL_SCRIPT`, `BM25_INDEX_PATH`, `CHROMA_PERSIST_DIR`,
   `CHROMA_COLLECTION`, `EMBEDDING_BACKEND`).
3. Construir el índice RAG (una vez, o cuando cambien los manuales):
   ```
   python -m rag.rag_index ./mis_manuales --bm25-output manual_bm25.pkl \
       --persist-dir ./chroma_manuales --collection manuales
   ```
4. Colocar un `.cfg`/`.dat` de ejemplo en `sample/` (o subir uno desde
   la UI).
5. `streamlit run app.py`

## Tests

Este repo no incluye actualmente una carpeta `tests/`. Como
verificación mínima de que el paquete importa sin ciclos, podés correr:

```
python -c "import app"
```

No reemplaza pruebas de integración contra un archivo COMTRADE real y
los 3 servidores MCP corriendo.
