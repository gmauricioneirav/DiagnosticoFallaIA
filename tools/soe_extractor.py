"""
soe_extractor.py

Extrae la Secuencia de Eventos (SOE - Sequence of Events) a partir de los
canales DIGITALES (de estado) de un archivo COMTRADE: arranques (pickup),
disparos (trip), posicion de interruptor, elementos de proteccion que
operaron, etc. -- ordenados cronologicamente con su timestamp exacto.

Esto es lo que despues alimenta `protection_status` en el
`FaultEventPayload` (event_schema.py), y tambien es evidencia clave para
que el agente de diagnostico entienda la secuencia real de actuacion de
las protecciones, no solo el estado final.

Un detalle importante descubierto al analizar datos reales: algunos
bits digitales llegan **constantes** durante todo el registro (ni suben
ni bajan) -- normalmente eso significa que es un bit de estado/latch de
un evento anterior, no una transicion real dentro de la ventana
capturada, y por eso NO deberia usarse como evidencia de que "ese
elemento opero durante este evento". Este modulo los identifica
explicitamente en vez de reportarlos como si fueran parte del SOE.

Requiere:
    pip install comtrade pandas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import comtrade


@dataclass
class SOEEvent:
    time_s: float                       # tiempo relativo al inicio del registro
    timestamp: datetime | None          # tiempo absoluto (start_timestamp + time_s), si esta disponible
    time_from_trigger_s: float | None   # tiempo relativo al disparo de la oscilografia (trigger_time)
    channel_name: str
    channel_index: int
    transition: str                 # "0->1" o "1->0"
    state_after: int


@dataclass
class SOEExtractor:
    cfg_path: str
    dat_path: str | None = None
    rec: comtrade.Comtrade = field(init=False, default=None)

    def load(self) -> "SOEExtractor":
        self.rec = comtrade.Comtrade()
        if self.dat_path:
            self.rec.load(self.cfg_path, self.dat_path)
        else:
            self.rec.load(self.cfg_path)
        return self

    # -- extraccion principal --------------------------------------------------

    def extract_soe(self, include_initial_state: bool = False) -> list[SOEEvent]:
        """
        Recorre cada canal digital y detecta TODAS las transiciones
        (0->1 y 1->0), devolviendo la lista completa ordenada
        cronologicamente (y, en caso de empate exacto de tiempo, por
        indice de canal, para que el orden sea determinista).

        `include_initial_state=True` agrega tambien un pseudo-evento en
        t=0 con el estado inicial de cada canal (util para saber, por
        ejemplo, si el interruptor ya estaba cerrado antes del evento) --
        no es una "transicion" real, se marca como tal en el campo
        `transition`.
        """
        if self.rec is None:
            self.load()

        status = np.array(self.rec.status)  # shape: (n_canales_digitales, n_muestras)
        time = np.array(self.rec.time)
        trigger_time = self.rec.trigger_time
        start_ts = self.rec.start_timestamp  # datetime del inicio del registro, si el CFG lo trae

        events: list[SOEEvent] = []

        for ch_idx, name in enumerate(self.rec.status_channel_ids):
            sig = status[ch_idx].astype(int)

            if include_initial_state:
                events.append(self._make_event(time[0], trigger_time, start_ts, name, ch_idx, "estado_inicial", int(sig[0])))

            diffs = np.diff(sig)
            change_indices = np.nonzero(diffs != 0)[0]
            for idx in change_indices:
                sample_idx = idx + 1  # el nuevo estado rige desde esta muestra
                new_state = int(sig[sample_idx])
                transition = "0->1" if new_state == 1 else "1->0"
                events.append(
                    self._make_event(time[sample_idx], trigger_time, start_ts, name, ch_idx, transition, new_state)
                )

        events.sort(key=lambda e: (e.time_s, e.channel_index))
        return events

    def _make_event(self, t: float, trigger_time: float | None, start_ts, name, ch_idx, transition, state) -> SOEEvent:
        timestamp = (start_ts + timedelta(seconds=float(t))) if isinstance(start_ts, datetime) else None
        rel_to_trigger = (t - trigger_time) if trigger_time is not None else None
        return SOEEvent(
            time_s=float(t), timestamp=timestamp, time_from_trigger_s=rel_to_trigger,
            channel_name=name, channel_index=ch_idx, transition=transition, state_after=state,
        )

    # -- canales que nunca transicionan (posible bit de estado/latch) ---------

    def channels_without_transitions(self) -> list[dict]:
        """
        Devuelve los canales digitales cuyo valor NO cambia en ningun
        momento del registro -- ni son parte de una transicion real
        capturada, y por eso no deberian tomarse como evidencia de "esto
        opero durante este evento" (ver nota del docstring del modulo).
        """
        if self.rec is None:
            self.load()

        status = np.array(self.rec.status)
        constant = []
        for ch_idx, name in enumerate(self.rec.status_channel_ids):
            sig = status[ch_idx]
            if np.all(sig == sig[0]):
                constant.append({"index": ch_idx, "name": name, "value": int(sig[0])})
        return constant

    # -- resumen compacto para el agente (nuevo) --------------------------------

    def summarize_protection_status(
        self, trigger_time: float | None = None, trip_channel_names: list[str] | None = None
    ) -> dict:
        """
        Reduce la secuencia completa de eventos a un resumen compacto y
        JSON-serializable, en el mismo espíritu que
        ComtradeFeatureExtractor.summarize_event: el agente recibe la
        evidencia de actuación de protecciones ya reducida, no la lista
        cruda de eventos ni las señales digitales.

        Solo se consideran activaciones (0->1) de canales que sí
        transicionan en algún punto del registro, ocurridas en o después
        del trigger -- los canales constantes (ver
        channels_without_transitions) se excluyen explícitamente porque
        no son evidencia de operación real en ESTE evento, y las
        transiciones 1->0 (reset) no se cuentan como "disparo".

        IMPORTANTE (descubierto probando con un registro real de un SEL-451):
        la PRIMERA activación tras el trigger casi nunca es el disparo real.
        En este archivo, la primera es "ER" (probablemente un flag interno
        de event recorder), que se activa y desactiva varias veces antes
        de que el relé dispare de verdad ("TRIP"/"3PT" varios ms después).
        Qué nombre de canal representa "el disparo" depende del fabricante
        y del esquema de protección configurado -- es conocimiento que
        debería salir del manual del relé (agente RAG), no de una
        heurística fija aquí. Por eso este método YA NO asume que la
        primera activación es el disparo: devuelve `first_activation_*`
        (neutral, siempre disponible) y solo llena `first_trip_*` si se
        le pasan `trip_channel_names` explícitos que matcheen algo.
        """
        if self.rec is None:
            self.load()
        trigger_time = trigger_time if trigger_time is not None else self.rec.trigger_time

        events = self.extract_soe(include_initial_state=False)
        constant = self.channels_without_transitions()
        constant_names = {c["name"] for c in constant}

        activations = [
            e for e in events
            if e.channel_name not in constant_names
            and e.transition == "0->1"
            and (trigger_time is None or e.time_s >= trigger_time)
        ]
        activations.sort(key=lambda e: e.time_s)

        first_activation = activations[0] if activations else None

        first_trip = None
        if trip_channel_names:
            wanted = {name.strip().upper() for name in trip_channel_names}
            trip_candidates = [e for e in activations if e.channel_name.strip().upper() in wanted]
            first_trip = trip_candidates[0] if trip_candidates else None

        return {
            "trigger_time_s": trigger_time,
            "first_activation_channel": first_activation.channel_name if first_activation else None,
            "first_activation_time_s": first_activation.time_s if first_activation else None,
            "trip_channel_names_used": list(trip_channel_names) if trip_channel_names else None,
            "trip_detected": (first_trip is not None) if trip_channel_names else None,
            "first_trip_channel": first_trip.channel_name if first_trip else None,
            "first_trip_time_s": first_trip.time_s if first_trip else None,
            "operating_time_ms": (
                (first_trip.time_s - trigger_time) * 1000.0
                if first_trip is not None and trigger_time is not None
                else None
            ),
            "activations_after_trigger": [
                {
                    "channel": e.channel_name,
                    "time_s": e.time_s,
                    "time_from_trigger_s": e.time_from_trigger_s,
                }
                for e in activations
            ],
            "constant_channels_excluded": [c["name"] for c in constant],
        }

    # -- salidas convenientes ---------------------------------------------------

    def to_dataframe(self, include_initial_state: bool = False) -> pd.DataFrame:
        events = self.extract_soe(include_initial_state=include_initial_state)
        return pd.DataFrame([
            {
                "time_s": e.time_s,
                "timestamp": e.timestamp,
                "time_from_trigger_s": e.time_from_trigger_s,
                "channel": e.channel_name,
                "channel_index": e.channel_index,
                "transition": e.transition,
                "state_after": e.state_after,
            }
            for e in events
        ])

    def report(self, include_initial_state: bool = False) -> str:
        """Texto plano legible, formato tipico de reporte SOE de un rele."""
        events = self.extract_soe(include_initial_state=include_initial_state)
        lines = [f"{'Tiempo (s)':>12}  {'t-trigger (s)':>14}  {'Transicion':<12}  Canal"]
        for e in events:
            t_trig = f"{e.time_from_trigger_s:+.5f}" if e.time_from_trigger_s is not None else "-"
            lines.append(f"{e.time_s:>12.5f}  {t_trig:>14}  {e.transition:<12}  {e.channel_name}")

        constants = self.channels_without_transitions()
        if constants:
            lines.append("")
            lines.append("Canales SIN transiciones durante el registro (posible bit de estado/latch,")
            lines.append("no usar como evidencia de operacion en tiempo real de este evento):")
            for c in constants:
                lines.append(f"  - {c['name']} (constante en {c['value']})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Uso de ejemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    soe = SOEExtractor(
        cfg_path=r"comtrades\sample3\oscilografia.CFG",
        dat_path=r"comtrades\sample3\oscilografia.DAT",
    ).load()

    print(soe.report())

    print("\n--- Como DataFrame (primeras filas) ---")
    df = soe.to_dataframe()
    print(df.head(10).to_string())

    print("\n--- Resumen compacto para el agente ---")
    print(json.dumps(soe.summarize_protection_status(), indent=2, default=str))
