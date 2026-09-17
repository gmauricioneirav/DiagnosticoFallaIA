"""
comtrade_features.py

Primera capa del pipeline de diagnostico de fallas electricas:
parseo de archivos COMTRADE + extraccion de features (RMS por ciclo
y componentes simetricas) listas para ser consumidas por los agentes
de analisis de senales y de diagnostico.

Esta capa es 100% deterministica (numpy/pandas) -- no involucra LLMs.
Los agentes reciben el DataFrame/resumen de salida, no las senales crudas.

Novedades de esta version respecto a la anterior:
  1. Usa ChannelIdentifier (channel_identifier.py) para localizar canales
     en vez de una tabla fija de nombres -- funciona con cualquier
     fabricante que ya soporte ese modulo.
  2. Prioriza las secuencias (I0/I1/I2, V0/V1/V2) y magnitudes fasoriales
     por fase YA CALCULADAS por el IED cuando estan disponibles, en vez
     de recalcularlas con DFT + Fortescue -- son la fuente de verdad que
     el propio rele uso para decidir si disparar. El calculo DFT queda
     como *fallback* para cuando el archivo no trae esas secuencias.
  3. Aplica correctamente la relacion primario/secundario del CT/TP
     (campos `primary`, `secondary`, `pors` del CFG) -- sin esto, las
     senales quedan en escala secundaria (ej. ~0.3A en vez de la
     corriente real de linea de cientos/miles de A).

Requiere:
    pip install comtrade numpy pandas
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import comtrade

from tools.channel_identifier import ChannelIdentifier, QuantityKind, Phase


# ---------------------------------------------------------------------------
# Utilidades de fasores y componentes simetricas (fallback DFT)
# ---------------------------------------------------------------------------

# Operador de rotacion de 120 grados usado en componentes simetricas
A_OP = complex(-0.5, math.sqrt(3) / 2)


def estimate_phasors(signal: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    """
    Estima el fasor de la componente fundamental para cada ciclo usando
    un filtro DFT de un ciclo (Full-Cycle DFT), la tecnica estandar en
    reles de proteccion numericos.

    Retorna un array de numeros complejos, uno por ciclo. La magnitud
    esta en valor PICO; se divide por sqrt(2) para obtener RMS.

    Convencion de angulo (importante para que Fortescue de resultados
    correctos): el fasor representa v(t) = Vm*cos(w*t + theta), como en
    la convencion estandar de ingenieria de potencia -- una senal que
    esta FISICAMENTE ATRASADA en el tiempo respecto a la referencia
    (coseno puro) debe tener theta NEGATIVO. Por eso la parte imaginaria
    se resta (`-imag`), no se suma: la proyeccion cruda contra seno da
    el signo opuesto al de esta convencion. Sin este signo, una fase B
    fisicamente atrasada 120 grados respecto a A sale con un fasor
    *adelantado* 120 grados, y la transformacion de Fortescue termina
    reportando la secuencia positiva como negativa y viceversa.
    """
    n_cycles = len(signal) // samples_per_cycle
    phasors = np.zeros(n_cycles, dtype=complex)

    k = np.arange(samples_per_cycle)
    cos_k = np.cos(2 * np.pi * k / samples_per_cycle)
    sin_k = np.sin(2 * np.pi * k / samples_per_cycle)

    for cycle in range(n_cycles):
        window = signal[cycle * samples_per_cycle: (cycle + 1) * samples_per_cycle]
        real = (2.0 / samples_per_cycle) * np.sum(window * cos_k)
        imag = (2.0 / samples_per_cycle) * np.sum(window * sin_k)
        phasors[cycle] = complex(real, -imag)  # signo corregido, ver docstring

    return phasors


def rms_per_cycle(signal: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    """RMS verdadero por ciclo (ventana no solapada)."""
    n_cycles = len(signal) // samples_per_cycle
    trimmed = signal[: n_cycles * samples_per_cycle]
    reshaped = trimmed.reshape(n_cycles, samples_per_cycle)
    return np.sqrt(np.mean(reshaped ** 2, axis=1))


def mean_per_cycle(signal: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    """
    Promedio simple por ciclo (no RMS). Se usa para leer canales que YA
    son una magnitud/angulo fasorial calculada por el IED -- promediar
    suaviza el ligero jitter de sample a sample sin distorsionar el valor,
    a diferencia de aplicar RMS a algo que no es una forma de onda oscilante.
    """
    n_cycles = len(signal) // samples_per_cycle
    trimmed = signal[: n_cycles * samples_per_cycle]
    reshaped = trimmed.reshape(n_cycles, samples_per_cycle)
    return np.mean(reshaped, axis=1)


def symmetrical_components(
    phasor_a: complex, phasor_b: complex, phasor_c: complex
) -> tuple[complex, complex, complex]:
    """
    Transformacion de Fortescue: dado un trio de fasores de fase (A, B, C)
    retorna (secuencia_cero, secuencia_positiva, secuencia_negativa).
    """
    zero = (phasor_a + phasor_b + phasor_c) / 3.0
    positive = (phasor_a + A_OP * phasor_b + A_OP ** 2 * phasor_c) / 3.0
    negative = (phasor_a + A_OP ** 2 * phasor_b + A_OP * phasor_c) / 3.0
    return zero, positive, negative


# ---------------------------------------------------------------------------
# Extractor principal
# ---------------------------------------------------------------------------

@dataclass
class ComtradeFeatureExtractor:
    cfg_path: str
    dat_path: str | None = None
    use_primary_values: bool = True   # convertir a valores de linea (primario) cuando el CFG lo permite
    winding_preference: int = 0       # indice a usar cuando hay varios canales candidatos por fase (0 = el primero)
    rec: comtrade.Comtrade = field(init=False, default=None)
    channel_map: dict = field(init=False, default=None)

    # -- carga -------------------------------------------------------------

    def load(self) -> "ComtradeFeatureExtractor":
        self.rec = comtrade.Comtrade()
        if self.dat_path:
            self.rec.load(self.cfg_path, self.dat_path)
        else:
            self.rec.load(self.cfg_path)

        # Se reutiliza el mismo objeto `rec` ya cargado para no releer el
        # archivo dos veces -- ChannelIdentifier acepta esto porque solo
        # llama a load() si su propio `rec` sigue en None.
        identifier = ChannelIdentifier(cfg_path=self.cfg_path, dat_path=self.dat_path)
        identifier.rec = self.rec
        self.channel_map = identifier.get_mapping()
        return self

    # -- utilidades de escala -----------------------------------------------

    def _primary_ratio(self, channel_index: int) -> float:
        """
        Retorna el factor por el que hay que multiplicar el valor ya
        convertido a unidades de ingenieria (rec.analog) para obtener el
        valor en el lado PRIMARIO (linea real), en vez del lado secundario
        del CT/TP.

        Si `pors` ya es 'P' (el fabricante calibro el multiplicador `a`
        directamente en primario), no se aplica ningun factor adicional
        -- aplicarlo de nuevo duplicaria el error.
        """
        if not self.use_primary_values:
            return 1.0
        chn = self.rec.cfg.analog_channels[channel_index]
        if str(chn.pors).strip().upper() == "P":
            return 1.0
        secondary = chn.secondary or 1.0
        primary = chn.primary or secondary  # si no hay dato, no se escala
        if secondary == 0:
            return 1.0
        return primary / secondary

    def _scaled_waveform(self, channel_index: int) -> np.ndarray:
        raw = np.array(self.rec.analog[channel_index], dtype=float)
        return raw * self._primary_ratio(channel_index)

    def _pick_index(self, candidates: list[int]) -> int | None:
        if not candidates:
            return None
        pos = self.winding_preference if self.winding_preference < len(candidates) else 0
        return candidates[pos]

    # -- extraccion de features ---------------------------------------------

    def extract(self, samples_per_cycle: int | None = None) -> pd.DataFrame:
        """
        Devuelve un DataFrame con una fila por ciclo:
          - RMS por fase (I y V), en primario si `use_primary_values=True`.
          - Componentes de secuencia (magnitud y angulo) para corriente y
            tension, tomadas de los canales precalculados por el IED
            cuando existen; si no, calculadas por DFT + Fortescue.
          - Columna `sequence_source` ("precomputed_ied" / "computed_dft")
            para que el agente sepa de donde salio el dato.
          - Factores de desbalance I2/I1 y V2/V1.
        """
        if self.rec is None:
            self.load()

        sample_rate = self.rec.cfg.sample_rates[0][0] if self.rec.cfg.sample_rates else None
        if not sample_rate:
            sample_rate = 1.0 / np.mean(np.diff(self.rec.time))
        nominal_freq = self.rec.cfg.frequency or 60.0
        spc = samples_per_cycle or round(sample_rate / nominal_freq)

        wf = self.channel_map["waveform"]
        current_wave_idx = {ph: self._pick_index(idxs) for ph, idxs in wf["current"].items()}
        voltage_wave_idx = {ph: self._pick_index(idxs) for ph, idxs in wf["voltage"].items()}

        n_cycles = len(self.rec.time) // spc
        cycle_time = np.array([self.rec.time[i * spc] for i in range(n_cycles)])

        # --- RMS por fase, siempre desde la forma de onda cruda ---
        current_rms = {
            ph: rms_per_cycle(self._scaled_waveform(idx), spc)
            for ph, idx in current_wave_idx.items() if idx is not None
        }
        voltage_rms = {
            ph: rms_per_cycle(self._scaled_waveform(idx), spc)
            for ph, idx in voltage_wave_idx.items() if idx is not None
        }

        # --- Secuencias: precalculadas por el IED si existen, si no DFT ---
        current_seq, current_seq_source = self._sequence_features("current", current_wave_idx, spc, n_cycles)
        voltage_seq, voltage_seq_source = self._sequence_features("voltage", voltage_wave_idx, spc, n_cycles)

        rows = []
        for cycle in range(n_cycles):
            row = {"cycle": cycle, "time_s": cycle_time[cycle]}

            for ph in ("a", "b", "c"):
                if ph in current_rms:
                    row[f"I{ph}_rms"] = current_rms[ph][cycle]
                if ph in voltage_rms:
                    row[f"V{ph}_rms"] = voltage_rms[ph][cycle]

            if current_seq is not None:
                row["I0_mag"] = current_seq["zero_mag"][cycle]
                row["I0_angle_deg"] = current_seq["zero_angle"][cycle]
                row["I1_mag"] = current_seq["pos_mag"][cycle]
                row["I1_angle_deg"] = current_seq["pos_angle"][cycle]
                row["I2_mag"] = current_seq["neg_mag"][cycle]
                row["I2_angle_deg"] = current_seq["neg_angle"][cycle]
                row["I_unbalance"] = row["I2_mag"] / row["I1_mag"] if row["I1_mag"] > 1e-9 else 0.0
                row["I_sequence_source"] = current_seq_source

            if voltage_seq is not None:
                row["V0_mag"] = voltage_seq["zero_mag"][cycle]
                row["V0_angle_deg"] = voltage_seq["zero_angle"][cycle]
                row["V1_mag"] = voltage_seq["pos_mag"][cycle]
                row["V1_angle_deg"] = voltage_seq["pos_angle"][cycle]
                row["V2_mag"] = voltage_seq["neg_mag"][cycle]
                row["V2_angle_deg"] = voltage_seq["neg_angle"][cycle]
                row["V_unbalance"] = row["V2_mag"] / row["V1_mag"] if row["V1_mag"] > 1e-9 else 0.0
                row["V_sequence_source"] = voltage_seq_source

            rows.append(row)

        return pd.DataFrame(rows)

    # -- secuencias: precalculadas o DFT -------------------------------------

    def _sequence_features(
        self, domain: str, waveform_idx: dict[str, int], spc: int, n_cycles: int
    ) -> tuple[dict | None, str]:
        """
        Intenta usar las secuencias precalculadas por el IED
        (channel_map['precomputed_sequence'][domain]); si no estan
        disponibles (falta alguna de las tres, o falta la magnitud),
        cae a calculo DFT + Fortescue desde las formas de onda crudas.

        Retorna (dict_de_arrays_por_ciclo, "precomputed_ied"|"computed_dft"),
        o (None, "unavailable") si no hay ni precalculado ni formas de onda
        de las 3 fases para calcular.
        """
        precomputed = self.channel_map["precomputed_sequence"].get(domain, {})
        has_full_precomputed = all(
            "magnitude" in precomputed.get(seq, {}) for seq in ("zero", "positive", "negative")
        )

        if has_full_precomputed:
            def mag_of(seq):
                idx = precomputed[seq]["magnitude"]
                return mean_per_cycle(self._scaled_waveform(idx), spc)[:n_cycles]

            def angle_of(seq):
                if "angle" not in precomputed[seq]:
                    return np.zeros(n_cycles)
                idx = precomputed[seq]["angle"]
                # el angulo no se reescala por primario/secundario (es en grados)
                raw = np.array(self.rec.analog[idx], dtype=float)
                return mean_per_cycle(raw, spc)[:n_cycles]

            result = {
                "zero_mag": mag_of("zero"), "zero_angle": angle_of("zero"),
                "pos_mag": mag_of("positive"), "pos_angle": angle_of("positive"),
                "neg_mag": mag_of("negative"), "neg_angle": angle_of("negative"),
            }
            return result, "precomputed_ied"

        # --- Fallback: calculo DFT + Fortescue desde formas de onda ---
        if not all(ph in waveform_idx and waveform_idx[ph] is not None for ph in ("a", "b", "c")):
            return None, "unavailable"

        phasors = {
            ph: estimate_phasors(self._scaled_waveform(waveform_idx[ph]), spc) for ph in ("a", "b", "c")
        }
        n = min(len(phasors["a"]), n_cycles)
        zero_mag = np.zeros(n); zero_angle = np.zeros(n)
        pos_mag = np.zeros(n); pos_angle = np.zeros(n)
        neg_mag = np.zeros(n); neg_angle = np.zeros(n)

        for cycle in range(n):
            z, p, ng = symmetrical_components(phasors["a"][cycle], phasors["b"][cycle], phasors["c"][cycle])
            zero_mag[cycle] = abs(z) / math.sqrt(2); zero_angle[cycle] = math.degrees(np.angle(z))
            pos_mag[cycle] = abs(p) / math.sqrt(2); pos_angle[cycle] = math.degrees(np.angle(p))
            neg_mag[cycle] = abs(ng) / math.sqrt(2); neg_angle[cycle] = math.degrees(np.angle(ng))

        result = {
            "zero_mag": zero_mag, "zero_angle": zero_angle,
            "pos_mag": pos_mag, "pos_angle": pos_angle,
            "neg_mag": neg_mag, "neg_angle": neg_angle,
        }
        return result, "computed_dft"

    # -- resumen para el agente ----------------------------------------------

    def summarize_event(self, df: pd.DataFrame, trigger_time: float | None = None) -> dict:
        """
        Reduce el DataFrame de ciclos a un resumen compacto (JSON-serializable)
        para el agente supervisor: contraste pre-falla vs ventana de falla,
        tomando la ventana de falla en el CICLO DE CORRIENTE PICO (no el
        maximo independiente de cada columna, que mezclaria instantes
        distintos y produciria una combinacion fisicamente inconsistente).
        """
        trigger_time = trigger_time if trigger_time is not None else self.rec.trigger_time
        
        pre_fault = df[df["time_s"] < (trigger_time - 0.02)] if trigger_time is not None else df.iloc[:1]
        during_fault = df[df["time_s"] >= trigger_time] if trigger_time is not None else df

        def safe_mean(frame, col):
            return float(frame[col].mean()) if col in frame.columns and not frame.empty else None

        summary = {
            "trigger_time_s": trigger_time,
            "prefault": {
                "Ia_rms": safe_mean(pre_fault, "Ia_rms"),
                "Ib_rms": safe_mean(pre_fault, "Ib_rms"),
                "Ic_rms": safe_mean(pre_fault, "Ic_rms"),
                "Va_rms": safe_mean(pre_fault, "Va_rms"),
                "Vb_rms": safe_mean(pre_fault, "Vb_rms"),
                "Vc_rms": safe_mean(pre_fault, "Vc_rms"),
            },
        }

        if during_fault.empty:
            summary["fault_window"] = None
            return summary

        current_cols = [c for c in ("Ia_rms", "Ib_rms", "Ic_rms") if c in during_fault.columns]
        peak_row = during_fault.loc[during_fault[current_cols].max(axis=1).idxmax()]

        summary["fault_window"] = {
            "peak_cycle_time_s": float(peak_row["time_s"]),
            "current_rms": {ph: float(peak_row.get(f"I{ph}_rms", float("nan"))) for ph in ("a", "b", "c")},
            "voltage_rms": {ph: float(peak_row.get(f"V{ph}_rms", float("nan"))) for ph in ("a", "b", "c")},
            "current_sequence": {
                "zero": {"magnitude": float(peak_row.get("I0_mag", float("nan"))), "angle_deg": float(peak_row.get("I0_angle_deg", float("nan")))},
                "positive": {"magnitude": float(peak_row.get("I1_mag", float("nan"))), "angle_deg": float(peak_row.get("I1_angle_deg", float("nan")))},
                "negative": {"magnitude": float(peak_row.get("I2_mag", float("nan"))), "angle_deg": float(peak_row.get("I2_angle_deg", float("nan")))},
                "unbalance_factor": float(peak_row.get("I_unbalance", float("nan"))),
                "source": peak_row.get("I_sequence_source"),
            },
            "voltage_sequence": {
                "zero": {"magnitude": float(peak_row.get("V0_mag", float("nan"))), "angle_deg": float(peak_row.get("V0_angle_deg", float("nan")))},
                "positive": {"magnitude": float(peak_row.get("V1_mag", float("nan"))), "angle_deg": float(peak_row.get("V1_angle_deg", float("nan")))},
                "negative": {"magnitude": float(peak_row.get("V2_mag", float("nan"))), "angle_deg": float(peak_row.get("V2_angle_deg", float("nan")))},
                "unbalance_factor": float(peak_row.get("V_unbalance", float("nan"))),
                "source": peak_row.get("V_sequence_source"),
            },
        }
        return summary


# ---------------------------------------------------------------------------
# Uso de ejemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("========= Archivo REAL (rele GE, con secuencias precalculadas) =========")
    real = ComtradeFeatureExtractor(
        cfg_path=r"comtrades\sample3\oscilografia.cfg",
        dat_path=r"comtrades\sample3\oscilografia.dat",
    ).load()
    df_real = real.extract()
    print(df_real[["cycle", "time_s", "Ia_rms", "Ib_rms", "Ic_rms", "I0_mag", "I1_mag", "I2_mag", "I_sequence_source"]].round(2).iloc[1:30].to_string())
    print()
    print(json.dumps(real.summarize_event(df_real), indent=2, default=str))

