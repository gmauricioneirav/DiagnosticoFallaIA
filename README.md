# FaultDiagnosisAI

Sistema multiagente (LangGraph + servidores MCP) para diagnóstico de
fallas eléctricas a partir de registros COMTRADE.

Este README documenta la organización de carpetas hecha sobre el proyecto:

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
│   └── report_agent.py         # Nodo "salida": orquesta signal_agent y redacta el informe
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
