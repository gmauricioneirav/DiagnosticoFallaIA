"""
app.py

Interfaz web (Streamlit) para el agente multiagente de diagnóstico de fallas
COMTRADE, orquestado con LangGraph + LangChain.

Principio de diseño: esta interfaz es una capa de PRESENTACIÓN PURA.
No contiene lógica de clasificación de fallas, ni cálculos de señales,
ni informes de respaldo. Todo el análisis -- ingesta, extracción de
features, consulta RAG, cálculos numéricos, diagnóstico, crítica y
generación de gráficas/informe -- ocurre dentro del grafo de agentes
(graph/workflow.py + agents/*.py) y de los servidores MCP que expone
(cálculo, graficado, retrieval -- ver mcp_servers/). La app solo invoca
el grafo y renderiza lo que este devuelve en su State.

No existe un "modo determinístico" de respaldo: si no hay agente
disponible (falta ANTHROPIC_API_KEY o los servidores MCP), la app lo
indica explícitamente en vez de mostrar un resultado calculado
localmente por reglas.

Flujo:
  1. Barra lateral: subir archivo .cfg/.dat (o usar el de ejemplo).
  2. Botón "Analizar evento": invoca el grafo LangGraph completo
     (Supervisor -> Ingesta -> Features -> RAG -> Diagnóstico -> Crítico
     -> Visualización/Informe -> ToolNode/MCP) y renderiza el State
     resultante.
  3. Chat: cada pregunta reinvoca el grafo sobre el mismo thread_id
     (vía checkpointer de LangGraph), preservando el State entre turnos.
     El Supervisor decide, en cada turno, si vuelve a Diagnóstico, a RAG,
     invoca una herramienta MCP, o responde directamente.

Contrato esperado del grafo (ver graph.workflow.build_graph):
  Entrada primer turno:  {"messages": [], "cfg_path": str, "dat_path": str}
  Entrada turnos de chat: {"messages": [{"role": "user", "content": str}]}
  Salida (State) debe incluir, cuando estén disponibles:
    - "diagnosis_hypothesis": str
    - "confidence": float (0-1) o str
    - "report_draft": str (markdown)
    - "figures": {
          "analog":  {"format": "png"|"svg"|"plotly_json", "data": str},
          "digital": {"format": "png"|"svg"|"plotly_json", "data": str},
      }
    - "messages": historial de mensajes (para extraer la respuesta del chat)

Backend de LLM (nube u modelo local), configurable sin tocar código
vía variable de entorno LLM_PROVIDER -- ver get_llm() más abajo:
    LLM_PROVIDER=openai (default)  -> requiere OPENAI_API_KEY
    LLM_PROVIDER=ollama            -> modelo local servido por Ollama

Ejecutar con:
    streamlit run app.py

    # con modelo local (requiere Ollama corriendo y el modelo descargado,
    # ver requirements.txt):
    LLM_PROVIDER=ollama OLLAMA_MODEL=qwen3:14b streamlit run app.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid

import streamlit as st

from graph.workflow import build_graph

from dotenv import load_dotenv
load_dotenv()


# Etiquetas legibles para mostrarle al usuario, en vivo, qué está haciendo
# el agente -- se usan al consumir los eventos de astream_events() en
# invoke_graph() más abajo. Las claves deben coincidir con los nombres de
# nodo registrados en graph.workflow.build_graph() y con los nombres de
# herramienta MCP (TOOL_* en graph/workflow.py).
NODE_LABELS = {
    "supervisor": "Decidiendo el siguiente paso...",
    "ingesta": "Realizando ingesta del archivo COMTRADE...",
    "features": "Analizando features (RMS, componentes simétricas, protecciones)...",
    "rag": "Consultando manuales normativos (RAG)...",
    "diagnostico": "Generando diagnóstico...",
    "critico": "Validando consistencia del diagnóstico...",
    "salida": "Generando gráficas e informe...",
}
TOOL_LABELS = {
    "extract_fault_features": "Utilizando herramienta MCP: extract_fault_features (cálculo)",
    "extract_protection_status": "Utilizando herramienta MCP: extract_protection_status (cálculo)",
    "retrieve_manuals": "Utilizando herramienta MCP: retrieve_manuals (retrieval)",
    "plot_signals": "Utilizando herramienta MCP: plot_signals (graficado)",
}

# ---------------------------------------------------------------------------
# Trazabilidad: logging en tiempo real + volcado del trace a disco
# ---------------------------------------------------------------------------

LOGS_DIR = os.environ.get("AGENT_LOGS_DIR", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def _configure_case_logging(thread_id: str) -> str:
    """
    Redirige a un archivo .txt, en tiempo real, los logs INFO que ya
    emite graph/workflow.py (logger.info en _traced() y en call_tool())
    -- cada paso del grafo y cada invocación de herramienta MCP, con
    timestamp, sin esperar a que termine el análisis. Devuelve la ruta.

    El logger se referencia por nombre "graph.workflow" (== __name__ de
    ese módulo, dado que vive en el paquete graph/) -- si en el futuro
    ese archivo se mueve o se renombra, este nombre debe actualizarse
    junto con él.
    """
    log_path = os.path.join(LOGS_DIR, f"case_{thread_id}.txt")
    target_logger = logging.getLogger("graph.workflow")
    target_logger.setLevel(logging.INFO)

    abs_path = os.path.abspath(log_path)
    for h in target_logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_path:
            return log_path  # ya configurado (evita duplicar handlers en reruns de Streamlit)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    target_logger.addHandler(handler)
    return log_path


def export_state_trace(state: dict, path: str) -> None:
    """
    Vuelca `state['trace']` (ver _traced() en graph/workflow.py) a un
    .txt como JSON indentado. A diferencia del log de texto libre de
    _configure_case_logging, esto captura de forma estructurada qué
    escribió cada nodo al State en cada paso -- listo para cargar con
    `json.load()` en un notebook/script y analizarlo cuantitativamente
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.get("trace", []), f, ensure_ascii=False, indent=2, default=str)


st.set_page_config(
    page_title="Agente IA para diagnóstico de fallas desde archivos COMTRADE",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cfg_path" not in st.session_state:
    st.session_state.cfg_path = None
if "dat_path" not in st.session_state:
    st.session_state.dat_path = None


def get_llm():
    """
    Selecciona el backend de LLM según la variable de entorno LLM_PROVIDER,
    para poder alternar entre nube y modelo local sin tocar código.

      LLM_PROVIDER=openai (default)
          Requiere OPENAI_API_KEY.
          OPENAI_MODEL (default: "gpt-5")

      LLM_PROVIDER=ollama
          Modelo local servido por Ollama (https://ollama.com).
          OLLAMA_MODEL     (default: "qwen3:14b")
          OLLAMA_BASE_URL  (default: "http://localhost:11434")

    IMPORTANTE: el modelo elegido -- de nube o local -- debe soportar
    salida estructurada (with_structured_output) y tool calling con
    fiabilidad razonable. El Supervisor, el Crítico y el agente ReAct de
    Diagnóstico dependen de eso para funcionar: una salida estructurada
    mal formada rompe el enrutamiento del grafo, no solo degrada una
    respuesta. Con Ollama, evita modelos <=8B para este sistema -- ver
    la nota en requirements.txt sobre qué modelos son aptos.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-5")
        return ChatOpenAI(model=model, temperature=0)

    raise ValueError(
        f"LLM_PROVIDER desconocido: {provider!r}. Usa 'openai' o 'ollama'."
    )


async def invoke_graph(
    cfg_path: str,
    dat_path: str,
    user_message: str | None = None,
    status: "st.delta_generator.DeltaGenerator | None" = None,
) -> dict:
    """
    Invoca (o continúa) el grafo de agentes sobre el thread_id de la sesión,
    transmitiendo en vivo -- si se pasa `status` (un objeto st.status()) --
    qué nodo y qué herramienta MCP está ejecutando el agente en cada
    momento.

    - Primer turno (user_message=None): se pasan las rutas de archivo;
      Ingesta, Features, RAG, Diagnóstico, Crítico y Visualización/Informe
      corren según decida el Supervisor (add_conditional_edges), no en un
      orden fijo.
    - Turnos de chat: se agrega el mensaje del usuario al mismo thread_id.
      El checkpointer de LangGraph recupera el State acumulado (features,
      resultados de cálculo, diagnóstico previo) para que el Supervisor
      decida con contexto completo si debe recalcular, volver a RAG, o
      responder directamente.

    Implementación: en vez de `app.ainvoke()` (que solo entrega el
    resultado final), se usa `app.astream_events()` para recibir cada
    evento de inicio/fin de nodo y de herramienta a medida que ocurren.
    El estado final consolidado no se arma a mano a partir de esos
    eventos -- se lee directamente del checkpointer con `aget_state()`,
    que es la fuente de verdad tras la ejecución (más robusto que confiar
    en la forma exacta del último evento, que puede variar entre
    versiones de langgraph/langchain-core).

    NOTA: los nombres de evento ("on_chain_start", "on_tool_start", etc.)
    y la presencia de `langgraph_node` en la metadata corresponden al
    esquema v2 de astream_events de langchain-core al momento de escribir
    esto. Si tras actualizar dependencias los pasos dejan de mostrarse en
    la UI, imprime `event` dentro del bucle para inspeccionar la forma
    real que está devolviendo tu versión instalada.
    """
    llm = get_llm()
    graph = build_graph(llm)
    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        # Margen adicional sobre el default de LangGraph (25) -- no es la
        # protección principal contra bucles (ver MAX_REVISION_CYCLES y
        # MAX_SUPERVISOR_STEPS en agents.coordinator.route_from_supervisor,
        # que fuerzan 'salida' mucho antes que esto), sino un respaldo por
        # si algún caso legítimo necesita más pasos de los previstos.
        "recursion_limit": 50,
        # No afecta la ejecución -- solo enriquece lo que LangSmith muestra
        # en el dashboard (si LANGSMITH_TRACING=true), para poder filtrar
        # y comparar corridas sin tener que abrir cada traza una por una.
        "run_name": "analisis_inicial" if user_message is None else "turno_chat",
        "tags": ["comtrade-diagnostico", os.environ.get("LLM_PROVIDER", "openai")],
        "metadata": {
            "thread_id": st.session_state.thread_id,
            "llm_provider": os.environ.get("LLM_PROVIDER", "openai"),
        },
    }

    if user_message is None:
        input_state = {"messages": [], "cfg_path": cfg_path, "dat_path": dat_path}
    else:
        input_state = {"messages": [{"role": "user", "content": user_message}]}

    if status is None:
        return await graph.ainvoke(input_state, config=config)

    seen_tools: set[str] = set()
    async for event in graph.astream_events(input_state, config=config, version="v2"):
        kind = event.get("event")
        name = event.get("name")

        if kind == "on_chain_start" and name in NODE_LABELS:
            status.update(label=NODE_LABELS[name])
            status.write(f"▶ {NODE_LABELS[name]}")
        elif kind == "on_tool_start" and name:
            label = TOOL_LABELS.get(name, f"Utilizando herramienta MCP: {name}")
            if name not in seen_tools:  # evita duplicar la línea si el ReAct de Diagnóstico repite la misma herramienta
                status.write(f"🔧 {label}")
            seen_tools.add(name)
        elif kind == "on_chain_end" and name in NODE_LABELS:
            status.write(f"✔ {NODE_LABELS[name]} — listo")

    snapshot = await graph.aget_state(config)
    return snapshot.values


# ---------------------------------------------------------------------------
# Barra lateral: carga de archivo
# ---------------------------------------------------------------------------

st.sidebar.title("Archivo COMTRADE")
uploaded_cfg = st.sidebar.file_uploader("Archivo .cfg", type=["cfg"])
uploaded_dat = st.sidebar.file_uploader("Archivo .dat", type=["dat"])
use_example = st.sidebar.checkbox("Usar archivo de ejemplo", value=not uploaded_cfg)

if uploaded_cfg and uploaded_dat and not use_example:
    tmp_dir = tempfile.mkdtemp()
    cfg_path = os.path.join(tmp_dir, uploaded_cfg.name)
    dat_path = os.path.join(tmp_dir, uploaded_dat.name)
    with open(cfg_path, "wb") as f:
        f.write(uploaded_cfg.getbuffer())
    with open(dat_path, "wb") as f:
        f.write(uploaded_dat.getbuffer())
else:
    # Antes vivía en comtrades/sample2/ -- ahora la carpeta de ejemplos
    # del proyecto es sample/ (ver la estructura de carpetas). Coloca ahí
    # tu propio .cfg/.dat de ejemplo con este mismo nombre/subcarpeta, o
    # ajusta esta ruta.
    cfg_path = os.path.join("sample", "sample2", "oscilografia.CFG")
    dat_path = os.path.join("sample", "sample2", "oscilografia.DAT")

run_button = st.sidebar.button("Analizar evento", type="primary")

with st.sidebar.expander("Sobre este agente"):
    st.caption(
        "Todo el análisis (extracción, cálculos, diagnóstico, gráficas e "
        "informe) lo realiza el grafo de agentes vía LangGraph, con los "
        "cálculos numéricos delegados a servidores MCP. Esta interfaz no "
        "ejecuta ninguna lógica de diagnóstico ni cálculo por su cuenta."
    )

_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

if _LLM_PROVIDER == "openai" and not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "No se encontró `OPENAI_API_KEY`. Con `LLM_PROVIDER=openai` (o sin "
        "definir la variable, que es el default) esta aplicación requiere "
        "esa credencial para el agente de IA; no existe un modo de informe "
        "o cálculo determinístico de respaldo. Configura la variable de "
        "entorno y recarga la página, o define `LLM_PROVIDER=ollama` para "
        "usar un modelo local en su lugar."
    )
    st.stop()
elif _LLM_PROVIDER == "ollama":
    st.sidebar.caption(
        f"Modelo local vía Ollama: `{os.environ.get('OLLAMA_MODEL', 'qwen3:14b')}` "
        f"en `{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}`. "
        f"Esta app no verifica que el servidor Ollama esté corriendo; si no "
        f"lo está, el error aparecerá al presionar \"Analizar evento\"."
    )
elif _LLM_PROVIDER not in ("openai", "ollama"):
    st.error(
        f"`LLM_PROVIDER={_LLM_PROVIDER!r}` no es válido. Usa `openai` u `ollama`."
    )
    st.stop()

if os.environ.get("LANGSMITH_TRACING", "").lower() == "true" or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
    st.sidebar.caption(f"🔍 Trazas activas en LangSmith (proyecto: `{os.environ.get('LANGSMITH_PROJECT', os.environ.get('LANGCHAIN_PROJECT', 'default'))}`).")


# ---------------------------------------------------------------------------
# Ejecutar el grafo de agentes
# ---------------------------------------------------------------------------

if run_button:
    st.session_state.cfg_path = cfg_path
    st.session_state.dat_path = dat_path
    st.session_state.thread_id = str(uuid.uuid4())  # nuevo evento -> nuevo hilo de checkpoint
    st.session_state.event_log_path = _configure_case_logging(st.session_state.thread_id)

    t0 = time.perf_counter()
    with st.status("Ejecutando el grafo de agentes...", expanded=True) as status:
        result = asyncio.run(invoke_graph(cfg_path, dat_path, status=status))
        status.update(label="Análisis completo", state="complete", expanded=False)
    st.session_state.analysis_duration_s = time.perf_counter() - t0

    st.session_state.graph_state = result
    st.session_state.trace_log_path = os.path.join(LOGS_DIR, f"case_{st.session_state.thread_id}_trace.txt")
    export_state_trace(result, st.session_state.trace_log_path)
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": result.get(
                "report_draft",
                "El agente aún no produjo un informe final; revisa el estado del grafo.",
            ),
        }
    ]

st.title("Agente IA para diagnóstico de fallas desde archivos COMTRADE")

if st.session_state.graph_state is None:
    st.info("Sube un archivo (o usa el de ejemplo) y presiona **Analizar evento** en la barra lateral.")
    st.stop()

state = st.session_state.graph_state


# ---------------------------------------------------------------------------
# Encabezado: diagnóstico y confianza tal como los produjo el agente
# ---------------------------------------------------------------------------

diagnosis = state.get("diagnosis_hypothesis")
confidence = state.get("confidence")

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(diagnosis or "Diagnóstico en proceso")
with col2:
    if confidence is not None:
        label = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)
        st.metric("Confianza del agente", label)

duration = st.session_state.get("analysis_duration_s")
trace = state.get("trace") or []
if duration is not None:
    st.caption(f"⏱ Tiempo total del análisis: {duration:.1f} s")
if trace:
    with st.expander(f"Paso a paso ejecutado por el agente ({len(trace)} pasos)"):
        for i, entry in enumerate(trace, start=1):
            step = entry.get("step", "?")
            step_duration = entry.get("duration_s")
            line = f"{i}. **{step}** — {step_duration:.2f} s" if step_duration is not None else f"{i}. **{step}**"
            st.markdown(line)
            if entry.get("rationale"):
                st.caption(f"↳ {entry['rationale']}")

        if st.session_state.get("event_log_path"):
            st.caption(f"Log en tiempo real: `{st.session_state.event_log_path}`")
        if st.session_state.get("trace_log_path") and os.path.exists(st.session_state.trace_log_path):
            with open(st.session_state.trace_log_path, "rb") as f:
                st.download_button(
                    "Descargar trace completo (.txt / JSON)",
                    data=f.read(),
                    file_name=os.path.basename(st.session_state.trace_log_path),
                    mime="text/plain",
                )


# ---------------------------------------------------------------------------
# Chat: cada turno reinvoca el grafo sobre el mismo thread_id
# ---------------------------------------------------------------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Preguntar sobre este evento (ej. '¿qué fase falló?', '¿cuánta corriente circuló?')"
)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    t0 = time.perf_counter()
    with st.status("El agente está pensando...", expanded=True) as status:
        result = asyncio.run(
            invoke_graph(st.session_state.cfg_path, st.session_state.dat_path, user_message=question, status=status)
        )
        status.update(label="Respuesta lista", state="complete", expanded=False)
    st.session_state.analysis_duration_s = time.perf_counter() - t0
    st.session_state.graph_state = result
    if st.session_state.get("trace_log_path"):
        export_state_trace(result, st.session_state.trace_log_path)

    answer = None
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        answer = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
    answer = answer or "El agente no devolvió una respuesta en este turno."

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


# ---------------------------------------------------------------------------
# Pestañas: señales analógicas y digitales, e informe -- todo generado
# por el agente (nodo Visualización/Informe -> ToolNode -> servidor MCP)
# ---------------------------------------------------------------------------

st.divider()
tab_analog, tab_digital, tab_report = st.tabs(["Señales analógicas", "Señales digitales", "Informe completo"])

figures = state.get("figures", {}) or {}


def render_agent_figure(fig_entry: dict | None, empty_msg: str) -> None:
    """
    Renderiza una figura producida por el agente. Nunca genera una gráfica
    localmente: si el agente todavía no la produjo, se muestra un mensaje
    en su lugar. Formatos soportados: 'png' (base64), 'svg' (markup),
    'plotly_json' (figura Plotly serializada).
    """
    if not fig_entry:
        st.info(empty_msg)
        return

    fmt = fig_entry.get("format")
    data = fig_entry.get("data")

    if fmt == "png":
        st.image(base64.b64decode(data))
    elif fmt == "svg":
        st.markdown(data, unsafe_allow_html=True)
    elif fmt == "plotly_json":
        import plotly.io as pio

        st.plotly_chart(pio.from_json(data), use_container_width=True)
    else:
        st.warning(f"Formato de figura no reconocido: {fmt!r}")


with tab_analog:
    render_agent_figure(
        figures.get("analog"),
        "El agente aún no ha generado la gráfica de señales analógicas.",
    )

with tab_digital:
    render_agent_figure(
        figures.get("digital"),
        "El agente aún no ha generado la gráfica de señales digitales.",
    )

with tab_report:
    st.markdown(state.get("report_draft", "Informe aún no generado por el agente."))
