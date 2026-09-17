# Asistente Inteligente Basado en LLM para el Diagnóstico Automático de Fallas Eléctricas en Sistemas de Potencia 

Sistema multiagente (LangGraph + servidores MCP + RAG) para 
diagnóstico de fallas eléctricas a partir de registros COMTRADE.

Este README documenta la organización de carpetas hecha sobre el proyecto:

## Estructura

```
DiagnosticoFallaIA/
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

## Recursos no incluidos en el repositorio

Algunos recursos necesarios para la ejecución completa de la aplicación no se incluyen directamente en el repositorio:

* Las bases de datos utilizadas por el sistema de recuperación mediante RAG no se incluyen en el repositorio debido a su tamaño. Estas incluyen la base de datos vectorial implementada mediante Chroma y el índice utilizado para la recuperación léxica mediante BM25.

* Los documentos técnicos, manuales y documentos corporativos utilizados para construir la base de conocimiento no se incluyen en el repositorio. Esta decisión responde a restricciones relacionadas con la confidencialidad y los derechos de uso de parte de la documentación técnica y corporativa utilizada como fuente de conocimiento.

Por tanto, para reproducir el funcionamiento del componente RAG es necesario construir nuevamente los índices a partir de los documentos técnicos disponibles para el usuario.

## Construcción de la base de conocimientos

Antes de ejecutar la aplicación es necesario disponer de los documentos técnicos que serán utilizados para construir la base de conocimiento del sistema.

Los documentos deben almacenarse en un directorio destinado a este propósito dentro de la estructura del proyecto. Para mantener la organización utilizada durante el desarrollo, se recomienda utilizar una carpeta denominada `Manuales` ubicada en el directorio raíz del proyecto.

Una vez disponibles los documentos, se ejecuta el proceso de construcción del índice RAG mediante el siguiente comando:

`python -m rag.rag_index Manuales --bm25-output manual_bm25.pkl --persist-dir ./chroma_manuales --collection manuales`

Este procedimiento genera los recursos necesarios para la recuperación de información utilizada por el sistema, incluyendo el índice BM25 y la base de datos vectorial persistente.

El proceso de construcción de los índices debe ejecutarse nuevamente cuando se realicen modificaciones en el conjunto documental, tales como la incorporación, eliminación o actualización de documentos utilizados como fuente de conocimiento.

## Instalación y ejecución de la aplicación

Una vez que el índice de conocimiento se encuentra disponible, es necesario instalar las dependencias requeridas por la aplicación.

Las librerías necesarias se encuentran especificadas en el archivo requirements.txt incluido en el repositorio. La instalación se realiza mediante el siguiente comando:

`pip install -r requirements.txt`

Posteriormente, se debe crear un archivo denominado `.env` en el directorio raíz del proyecto. El repositorio incluye un archivo de ejemplo denominado `.env.example`, que puede utilizarse como referencia para la definición de las variables de entorno requeridas. 

Como mínimo, debe configurarse la variable correspondiente a la clave de acceso al servicio de modelos de lenguaje:

`OPENAI_API_KEY=...`

Las demás variables de configuración pueden utilizar los valores predeterminados definidos por la aplicación cuando no se especifiquen explícitamente en el archivo `.env`.

Finalmente, la aplicación puede ejecutarse mediante Streamlit utilizando el siguiente comando: 

`streamlit run app.py`

La ejecución de este comando inicia la aplicación y permite acceder a la interfaz desde un navegador web.


## Consideraciones y limitaciones para la reproducibilidad

La reproducibilidad del código y de la arquitectura desarrollada se facilita mediante la disponibilidad del repositorio y de los registros utilizados en la evaluación. Sin embargo, la reproducción exacta de los resultados asociados al componente RAG puede verse limitada por la imposibilidad de distribuir parte de la documentación técnica utilizada durante su construcción.

La separación entre el código fuente y los recursos documentales permite mantener disponibles los componentes desarrollados durante la investigación, respetando simultáneamente las restricciones asociadas a la documentación técnica y corporativa empleada.

La disponibilidad del código fuente en el repositorio permite revisar la implementación de los principales componentes del sistema, incluyendo la arquitectura multiagente, los mecanismos de orquestación, las herramientas utilizadas, el sistema de recuperación de información y la interfaz de usuario.
