"""
mcp_servers/graficado_server.py

Servidor MCP de graficado para el agente de diagnóstico de fallas
COMTRADE. Lógica de dibujo: pestañas de señales analógicas/digitales con matplotlib,
Servidor MCP: el agente (agents/signal_agent.py, invocado desde
agents/report_agent.py) solo pasa cfg_path/dat_path y recibe una imagen
ya lista para mostrar. Ninguna señal cruza la frontera del servidor
como argumento, y el frontend (app.py) nunca dibuja nada por su cuenta.

Herramienta expuesta:
  - plot_signals(cfg_path, dat_path, domain, fault_window=None, phases=None)
      domain="analog"  -> corriente y tensión por fase, vía
                          ComtradeFeatureExtractor (tools/comtrade_features.py)
      domain="digital" -> secuencia de estados de canales digitales, vía
                          SOEExtractor (tools/soe_extractor.py)

    Devuelve {"format": "png", "data": "<base64>"} -- exactamente el
    formato que espera render_agent_figure() en app.py.

NOTA de diseño: se reutilizan extractor._pick_index() y
extractor._scaled_waveform(), que en tools/comtrade_features.py están
marcados como "privados" por convención (guión bajo simple). Como este
servidor es parte del mismo proyecto, se acepta por ahora; si más
adelante otro equipo va a mantener el servidor de graficado por
separado, conviene promoverlos a métodos públicos en
ComtradeFeatureExtractor.

Ejecutar:
    python mcp_servers/graficado_server.py
    MCP_TRANSPORT=http MCP_PORT=8002 python mcp_servers/graficado_server.py

Requiere:
    pip install "mcp[cli]" comtrade numpy pandas matplotlib
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional

# Ver la nota equivalente en mcp_servers/calculo_server.py: sin esto,
# `from tools.comtrade_features import ...` falla dentro del subproceso
# porque sys.path[0] queda apuntando a mcp_servers/, no a la raíz del
# proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # backend sin display -- imprescindible en un servidor
import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP

from tools.comtrade_features import ComtradeFeatureExtractor
from tools.soe_extractor import SOEExtractor

mcp = FastMCP(
    "graficado",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8002")),
)

_PHASE_COLORS = {"a": "tab:orange", "b": "tab:cyan", "c": "tab:pink"}


def _encode_figure(fig) -> str:
    """Serializa una figura de matplotlib a PNG en base64 (sin prefijo
    data URI -- app.py hace base64.b64decode(data) directamente)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_analog(cfg_path: str, dat_path: str, fault_window: Optional[dict], phases: Optional[list[str]]) -> dict:
    phases = phases or ["a", "b", "c"]
    extractor = ComtradeFeatureExtractor(cfg_path=cfg_path, dat_path=dat_path).load()
    time = extractor.rec.time
    wf = extractor.channel_map["waveform"]

    fig, (ax_i, ax_v) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ph in phases:
        color = _PHASE_COLORS.get(ph)

        idx_i = extractor._pick_index(wf["current"].get(ph, []))
        if idx_i is not None:
            ax_i.plot(time, extractor._scaled_waveform(idx_i), label=f"I{ph.upper()}", color=color, linewidth=0.9)

        idx_v = extractor._pick_index(wf["voltage"].get(ph, []))
        if idx_v is not None:
            ax_v.plot(time, extractor._scaled_waveform(idx_v), label=f"V{ph.upper()}", color=color, linewidth=0.9)

    # Solo tenemos el ciclo de PICO de la falla (fault_window.peak_cycle_time_s
    # en el resumen de ComtradeFeatureExtractor.summarize_event), no un
    # rango de inicio/fin -- por eso se marca con una línea vertical y no
    # con una banda de "ventana", que implicaría un dato que no tenemos.
    peak_t = (fault_window or {}).get("peak_cycle_time_s")
    if peak_t is not None:
        for ax in (ax_i, ax_v):
            ax.axvline(peak_t, color="red", linestyle="--", linewidth=1, alpha=0.7, label="pico de falla")

    ax_i.set_ylabel("Corriente (A)")
    ax_i.legend(loc="upper right", fontsize=8)
    ax_i.grid(alpha=0.2)
    ax_v.set_ylabel("Tensión (V)")
    ax_v.set_xlabel("Tiempo (s)")
    ax_v.legend(loc="upper right", fontsize=8)
    ax_v.grid(alpha=0.2)
    fig.tight_layout()

    return {"format": "png", "data": _encode_figure(fig)}


from collections import defaultdict


def _plot_digital(
    cfg_path: str,
    dat_path: str,
    fault_window: Optional[dict],
) -> dict:

    soe = SOEExtractor(
        cfg_path=cfg_path,
        dat_path=dat_path,
    ).load()

    events = soe.extract_soe(include_initial_state=False)

    constant_channels = soe.channels_without_transitions()

    time = soe.rec.time

    xmin = time[0]
    xmax = time[-1]

    # --------------------------------------------------------
    # Agrupar eventos por canal (O(N))
    # --------------------------------------------------------

    channel_events = defaultdict(list)

    channel_order = []

    for e in events:

        if not channel_events[e.channel_name]:
            channel_order.append(e.channel_name)

        channel_events[e.channel_name].append(e)

    channel_order.reverse()

    if not channel_order:

        fig, ax = plt.subplots(figsize=(10, 1.5))

        ax.text(
            0.5,
            0.5,
            "No se detectaron transiciones en los canales digitales",
            ha="center",
            va="center",
        )

        ax.axis("off")

        return {
            "format": "png",
            "data": _encode_figure(fig),
        }

    # --------------------------------------------------------
    # Parámetros gráficos
    # --------------------------------------------------------

    CHANNEL_HEIGHT = 1.60

    LOW = 0.25

    HIGH = 1.25

    fig_height = max(
        4,
        len(channel_order) * 0.55,
    )

    fig, ax = plt.subplots(
        figsize=(14, fig_height),
        constrained_layout=True,
    )

    yticks = []

    ylabels = []

    x_offset = (xmax - xmin) * 0.004

    # --------------------------------------------------------
    # Dibujar canales
    # --------------------------------------------------------

    for row, channel in enumerate(channel_order):

        base = row * CHANNEL_HEIGHT

        ch_events = channel_events[channel]

        initial = 1 - ch_events[0].state_after

        xs = [xmin]

        ys = [base + (HIGH if initial else LOW)]

        current = initial

        for ev in ch_events:

            level = base + (HIGH if current else LOW)

            xs.extend([ev.time_s, ev.time_s])

            ys.extend([level])

            current = ev.state_after

            ys.extend([
                base + (HIGH if current else LOW)
            ])

        xs.append(xmax)

        ys.append(
            base + (HIGH if current else LOW)
        )

        ax.plot(
            xs,
            ys,
            color="tab:green",
            linewidth=1.6,
        )

        # Separador

        ax.axhline(
            base,
            color="0.90",
            linewidth=0.7,
            zorder=0,
        )

        # Estados lógicos

        ax.text(
            xmin - x_offset,
            base + LOW,
            "0",
            fontsize=7,
            ha="right",
            va="center",
            color="0.45",
        )

        ax.text(
            xmin - x_offset,
            base + HIGH,
            "1",
            fontsize=7,
            ha="right",
            va="center",
            color="0.45",
        )

        yticks.append(base + CHANNEL_HEIGHT / 2)

        ylabels.append(channel)

    ax.axhline(
        len(channel_order) * CHANNEL_HEIGHT,
        color="0.90",
        linewidth=0.7,
    )

    # --------------------------------------------------------
    # Pico de falla
    # --------------------------------------------------------

    peak = (
        fault_window or {}
    ).get("peak_cycle_time_s")

    if peak is not None:

        ax.axvline(
            peak,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

    # --------------------------------------------------------
    # Configuración
    # --------------------------------------------------------

    ax.set_xlim(
        xmin - x_offset * 12,
        xmax,
    )

    ax.set_ylim(
        0,
        len(channel_order) * CHANNEL_HEIGHT,
    )

    ax.set_yticks(yticks)

    ax.set_yticklabels(
        ylabels,
        fontsize=8,
        fontfamily="monospace",
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=3,
    )

    ax.set_xlabel("Tiempo (s)")

    ax.set_ylabel(
        "Canales digitales",
        fontsize=10,
        labelpad=4,
    )

    ax.grid(
        axis="x",
        color="0.88",
    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    if constant_channels:

        fig.text(
            0.01,
            0.01,
            f"{len(constant_channels)} canales sin transición "
            "(ver extract_protection_status).",
            fontsize=7,
        )

    return {
        "format": "png",
        "data": _encode_figure(fig),
    }

@mcp.tool()
async def plot_signals(
    cfg_path: str,
    dat_path: str,
    domain: str,
    fault_window: Optional[dict] = None,
    phases: Optional[list[str]] = None,
) -> dict:
    """Genera la gráfica de señales de un registro COMTRADE.

    Args:
        cfg_path: ruta al archivo .cfg del registro COMTRADE.
        dat_path: ruta al archivo .dat del registro COMTRADE.
        domain: "analog" (corriente y tensión por fase) o "digital"
            (secuencia de estados de canales digitales).
        fault_window: opcionalmente, el sub-dict "fault_window" que
            devuelve extract_fault_features (con "peak_cycle_time_s"),
            para marcar el pico de la falla en la gráfica.
        phases: lista de fases a graficar en dominio "analog"
            (por defecto ["a", "b", "c"]). Ignorado en dominio "digital".

    Returns:
        {"format": "png", "data": "<base64>"}
    """
    if domain not in ("analog", "digital"):
        raise ValueError(f"domain debe ser 'analog' o 'digital', recibido: {domain!r}")

    def _run():
        if domain == "analog":
            return _plot_analog(cfg_path, dat_path, fault_window, phases)
        return _plot_digital(cfg_path, dat_path, fault_window)

    return await asyncio.to_thread(_run)


if __name__ == "__main__":
    transport = "streamable-http" if os.environ.get("MCP_TRANSPORT") == "http" else "stdio"
    mcp.run(transport=transport)
