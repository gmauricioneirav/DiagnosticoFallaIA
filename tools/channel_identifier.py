"""
channel_identifier.py

Identifica automáticamente, dentro de un archivo COMTRADE, qué canales
analógicos corresponden a corriente y cuáles a tensión, y a qué fase
(A, B, C, neutro/residual) pertenece cada uno — independientemente del
fabricante del relé/registrador (ABB, Siemens, SEL, GE, etc.), cada uno
con su propia convención de nomenclatura de canales.

Estrategia de identificación, en orden de confiabilidad:
  1. Unidad del canal (campo `uu` del CFG: A, kA, V, kV) -> tipo de señal.
  2. Campo `ph` del CFG (cuando el fabricante lo llena según el estándar
     IEEE C37.111) -> fase directa.
  3. Si `ph` viene vacío (muy común en la práctica), se infiere la fase
     por patrones de nombre (regex) que cubren las convenciones más
     usadas: IA/IB/IC, IL1/IL2/IL3, IR/IS/IT, VAN/VBN/VCN, VAB/VBC/VCA,
     UL1/UL2/UL3, corrientes/residuales IN/IG/IE/3I0, etc. Estos
     patrones son ANCLADOS (deben calzar con el nombre completo, ya
     normalizado) para evitar falsos positivos.
  3.5. Si lo anterior no calza (nombre completo no es un token corto
     tipo "IA"), se prueba un patrón de SUFIJO de menor confianza sobre
     el nombre CRUDO: exportadores como DIgSILENT PowerFactory anteponen
     el nombre de la subestación/línea antes de una descripción tipo
     "Phase Current A" o "3*I0" (ej. "Piedecuesta - Rio Frio 115:Phase
     Current A"), con lo que ningún patrón anclado del punto 3 calza
     contra el nombre completo. Este paso busca el token de fase (A/B/C)
     o el patrón residual (3*I0/3I0) solo al FINAL del nombre, precedido
     de un separador (espacio, ":", "_", "-") o inicio de cadena.
  4. Antes de rendirse, se revisa si el canal es una **cantidad fasorial
     ya calculada por el relé** (muy común en GE, SEL y otros IEDs
     modernos): magnitud/ángulo de secuencia cero/positiva/negativa
     (ej. "LINEA I_0 Mag", "I_1 Angle", "V_2 Mag") o magnitud fasorial
     por fase (ej. "LINEA Ia Mag"), o la frecuencia de seguimiento
     ("Tracking Frequency"). Estas NO son formas de onda muestreadas,
     son valores fasoriales/escalares que el propio IED ya calculó —
     y por eso NO deben pasarse al extractor de features como si
     fueran una señal cruda de fase.
  5. Si nada calza, el canal queda como "no clasificado" para revisión
     manual — mejor eso que asignar una fase incorrecta en silencio.

Requiere:
    pip install comtrade
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import comtrade


class SignalType(str, Enum):
    CURRENT = "current"
    VOLTAGE = "voltage"
    POTENCIA_ACTIVA = "potencia activa"
    POTENCIA_REACTIVA = "potencia reactiva"
    UNKNOWN = "unknown"


class Phase(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    NEUTRAL = "N"      # residual/neutro (3I0, 3V0, IN, VN, IG, VG...)
    UNKNOWN = "unknown"


class QuantityKind(str, Enum):
    """
    Distingue una forma de onda muestreada (lo que necesita el
    extractor de RMS/DFT) de una cantidad ya derivada que el propio
    IED calculó y expone como canal analógico adicional.
    """
    WAVEFORM = "waveform"                    # muestra instantánea de fase (IA, VB, IL1, etc.)
    SEQUENCE_ZERO = "sequence_zero"          # I_0 / V_0 precalculado
    SEQUENCE_POSITIVE = "sequence_positive"  # I_1 / V_1 precalculado
    SEQUENCE_NEGATIVE = "sequence_negative"  # I_2 / V_2 precalculado
    PHASOR_PHASE = "phasor_phase"            # magnitud/ángulo fasorial por fase ya calculada (ej. "Ia Mag")
    FREQUENCY = "frequency"                  # frecuencia de seguimiento del sistema
    UNKNOWN = "unknown"


class PhasorPart(str, Enum):
    MAGNITUDE = "magnitude"
    ANGLE = "angle"
    NONE = "none"        # no aplica (es una forma de onda, no un fasor precalculado)


@dataclass
class ChannelInfo:
    index: int
    name: str
    unit: str
    signal_type: SignalType
    phase: Phase
    connection: str = ""     # "wye" (fase-neutro) o "delta" (fase-fase), si se puede inferir
    confidence: str = ""     # de dónde salió la clasificación, para auditoría
    quantity_kind: QuantityKind = QuantityKind.WAVEFORM
    phasor_part: PhasorPart = PhasorPart.NONE
    domain_hint: str | None = None   # "current"/"voltage" inferido del nombre, usado cuando uu no ayuda (ej. ángulos en "deg")


# ---------------------------------------------------------------------------
# Unidades -> tipo de señal
# ---------------------------------------------------------------------------

CURRENT_UNITS = {"A", "KA", "MA"}
VOLTAGE_UNITS = {"V", "KV", "MV"}
POTENCIA_ACTIVA_UNITS = {"W", "KW", "MW"}
POTENCIA_REACTIVA_UNITS = {"VAR", "KVAR", "MVAR"}


def classify_by_unit(unit: str) -> SignalType:
    u = unit.strip().upper()
    if u in CURRENT_UNITS:
        return SignalType.CURRENT
    if u in VOLTAGE_UNITS:
        return SignalType.VOLTAGE
    if u in POTENCIA_ACTIVA_UNITS:
        return SignalType.POTENCIA_ACTIVA
    if u in POTENCIA_REACTIVA_UNITS:
        return SignalType.POTENCIA_REACTIVA
    return SignalType.UNKNOWN


# ---------------------------------------------------------------------------
# Patrones de nombre por fabricante/convención
# ---------------------------------------------------------------------------
# Cada patrón se evalúa contra el nombre del canal en mayúsculas y sin
# espacios. Se agrupan por fase; el tipo (I/V) ya se resolvió por unidad,
# así que aquí solo se busca la fase.

PHASE_PATTERNS: dict[Phase, list[re.Pattern]] = {
    Phase.A: [
        re.compile(r"^[IUV]?A[N1]?$"),            # IA, VA, UA, IA1, VAN, IAN
        re.compile(r"^[IUV]?L1$"),                # IL1, VL1, UL1 (IEC/Siemens)
        re.compile(r"^LINE_[IUV]?L1$"),           # IL1, VL1, UL1 (IEC/Siemens)
        re.compile(r"^[IUV]?R$"),                 # IR, VR (convención R-S-T)
        re.compile(r"^[IUV]?AW\d*$"),             # IAW, IAW1 (SEL, bobina W)
        re.compile(r"^[IUV]?AX\d*$"),             # IAX (SEL, bobina X / segundo devanado)
        re.compile(r"^[IUV]?A[GN]$"),             # VAG, IAG (referencia a tierra explícita)
    ],
    Phase.B: [
        re.compile(r"^[IUV]?B[N1]?$"),
        re.compile(r"^[IUV]?L2$"),    
        re.compile(r"^LINE_[IUV]?L2$"),                
        re.compile(r"^[IUV]?S$"),
        re.compile(r"^[IUV]?BW\d*$"),
        re.compile(r"^[IUV]?BX\d*$"),
        re.compile(r"^[IUV]?B[GN]$"),
    ],
    Phase.C: [
        re.compile(r"^[IUV]?C[N1]?$"),
        re.compile(r"^[IUV]?L3$"),     
        re.compile(r"^LINE_[IUV]?L3$"),             
        re.compile(r"^[IUV]?T$"),
        re.compile(r"^[IUV]?CW\d*$"),
        re.compile(r"^[IUV]?CX\d*$"),
        re.compile(r"^[IUV]?C[GN]$"),
    ],
    Phase.NEUTRAL: [
        re.compile(r"^[IUV]?N$"),                 # IN, VN
        re.compile(r"^LINE_[IUV]?N$"),    
        re.compile(r"^[IUV]?G$"),                 # IG, VG (ground)
        re.compile(r"^[IUV]?E$"),                 # IE, UE (Siemens: earth)
        re.compile(r"^3[IUV]0$"),                 # 3I0, 3V0 (residual clásico)
        re.compile(r"^[IUV]?RES(IDUAL)?$"),
        re.compile(r"^[IUV]?0$"),                 # I0, V0 (secuencia cero directa, si viene precalculada)
    ],
}

# Pares fase-fase (delta): si el nombre matchea esto, es una tensión
# línea-línea, no línea-neutro. Se le asigna la fase "líder" del par.
DELTA_PATTERNS: dict[Phase, re.Pattern] = {
    Phase.A: re.compile(r"^[UV]?(AB|A_B|L1L2|L12)$"),
    Phase.B: re.compile(r"^[UV]?(BC|B_C|L2L3|L23)$"),
    Phase.C: re.compile(r"^[UV]?(CA|C_A|L3L1|L31)$"),
}


def _normalize(name: str) -> str:
    return re.sub(r"[\s\-]", "", name.strip().upper())


def classify_phase_by_name(raw_name: str) -> tuple[Phase, str]:
    """Retorna (fase, tipo_conexión) inferidos del nombre del canal."""
    name = _normalize(raw_name)

    for phase, pattern in DELTA_PATTERNS.items():
        if pattern.match(name):
            return phase, "delta"

    for phase, patterns in PHASE_PATTERNS.items():
        for pattern in patterns:
            if pattern.match(name):
                connection = "wye" if phase != Phase.NEUTRAL else "residual"
                return phase, connection

    return Phase.UNKNOWN, ""


# ---------------------------------------------------------------------------
# Fallback de sufijo: exportadores con nombre de estación/línea antepuesto
# ---------------------------------------------------------------------------
# Los patrones de PHASE_PATTERNS son anclados (^...$) y exigen que el
# nombre COMPLETO, ya normalizado, sea un token corto tipo "IA" o "VAN" --
# lo que evita falsos positivos, pero no calza con exportadores que anteponen
# un nombre de estación/línea largo antes de una descripción de fase, p. ej.
# DIgSILENT PowerFactory: "Piedecuesta - Rio Frio 115:Phase Current A".
#
# Este fallback opera sobre el nombre CRUDO (antes de quitar espacios) y
# solo mira el FINAL de la cadena: un token de fase aislado (precedido de
# espacio, ":", "_", "-" o inicio de cadena) o el patrón residual clásico
# "3I0"/"3*I0". Es deliberadamente de MENOR confianza que un match de
# PHASE_PATTERNS -- se etiqueta con su propio valor de `confidence` para
# que quede auditable -- porque un canal genuinamente llamado, por ejemplo,
# "...Bahía A" (una bahía/posición de subestación, no una fase eléctrica)
# también haría match. En la práctica, esto es preferible a dejar sin
# clasificar un canal cuya UNIDAD (uu) ya confirmó que es corriente o
# tensión: se prioriza no perder la señal sobre el riesgo, acotado y
# auditable, de una fase mal asignada por este último recurso.

_SUFFIX_PHASE_PATTERN = re.compile(r"(?:^|[\s:_-])([ABC])$", re.IGNORECASE)
_SUFFIX_RESIDUAL_PATTERN = re.compile(r"3\*?[IUV]0$", re.IGNORECASE)


def classify_phase_by_suffix(raw_name: str) -> tuple[Phase, str]:
    """Fallback de menor confianza: token de fase o residual al final
    del nombre CRUDO (ver nota arriba)."""
    name = raw_name.strip()

    if _SUFFIX_RESIDUAL_PATTERN.search(name):
        return Phase.NEUTRAL, "residual"

    match = _SUFFIX_PHASE_PATTERN.search(name)
    if match:
        phase = {"A": Phase.A, "B": Phase.B, "C": Phase.C}[match.group(1).upper()]
        return phase, "wye"

    return Phase.UNKNOWN, ""


# ---------------------------------------------------------------------------
# Cantidades fasoriales/derivadas precalculadas por el IED
# ---------------------------------------------------------------------------
# Muchos relés modernos (GE, SEL, entre otros) exponen, además de las
# formas de onda crudas, canales analógicos ya derivados: magnitud y
# ángulo de secuencia (I_0, I_1, I_2, V_0, V_1, V_2), magnitud fasorial
# por fase (Ia Mag, Ib Mag, Ic Mag) y la frecuencia de seguimiento.
# Estos canales suelen venir con nombres largos con prefijo del
# elemento o línea (ej. "LINEA  I_0 Mag"), por lo que aquí se usa
# búsqueda de subcadena (search) en vez de match anclado.

_MAG_SUFFIX = re.compile(r"MAG$")
_ANGLE_SUFFIX = re.compile(r"ANGLE$")

# Secuencia: se busca el token I_0/I_1/I_2/V_0/V_1/V_2 en cualquier parte
# del nombre normalizado (que conserva guiones bajos).
_SEQUENCE_TOKEN_PATTERNS: dict[tuple[str, QuantityKind], re.Pattern] = {
    ("I", QuantityKind.SEQUENCE_ZERO): re.compile(r"I_?0(?!\d)"),
    ("I", QuantityKind.SEQUENCE_POSITIVE): re.compile(r"I_?1(?!\d)"),
    ("I", QuantityKind.SEQUENCE_NEGATIVE): re.compile(r"I_?2(?!\d)"),
    ("V", QuantityKind.SEQUENCE_ZERO): re.compile(r"V_?0(?!\d)"),
    ("V", QuantityKind.SEQUENCE_POSITIVE): re.compile(r"V_?1(?!\d)"),
    ("V", QuantityKind.SEQUENCE_NEGATIVE): re.compile(r"V_?2(?!\d)"),
}

# Magnitud fasorial por fase ya calculada (distinta de la forma de onda
# cruda): "Ia Mag", "Ib Angle", "Vc Mag", etc. Se exige que el token de
# fase (a/b/c) quede pegado inmediatamente antes del sufijo Mag/Angle,
# para no confundir con "3Ia..." u otras variantes.
_PHASOR_PHASE_PATTERNS: dict[Phase, re.Pattern] = {
    Phase.A: re.compile(r"[IV]A(?:MAG|ANGLE)$"),
    Phase.B: re.compile(r"[IV]B(?:MAG|ANGLE)$"),
    Phase.C: re.compile(r"[IV]C(?:MAG|ANGLE)$"),
}

_FREQUENCY_PATTERN = re.compile(r"FREQUENC")


_PHASOR_PHASE_DOMAIN: dict[Phase, str] = {}  # se completa dinámicamente al matchear (I->current, V->voltage)


def classify_precomputed_quantity(raw_name: str) -> tuple[QuantityKind, Phase, PhasorPart, str | None]:
    """
    Detecta si un canal es una cantidad ya derivada por el IED (secuencia,
    magnitud fasorial por fase, o frecuencia) en vez de una forma de onda
    cruda. Retorna (QuantityKind.UNKNOWN, Phase.UNKNOWN, PhasorPart.NONE, None)
    si no aplica ninguno de estos patrones.

    El cuarto elemento retornado ("domain") indica si la cantidad es de
    corriente o tensión ("current"/"voltage"), inferido del token I_/V_
    o del prefijo I/V en el nombre — esto es necesario porque los canales
    de **ángulo** vienen con unidad "deg" en el CFG, no "A" ni "V", así
    que no se puede confiar en `chn.uu` para saber a qué dominio pertenecen.
    """
    name = _normalize(raw_name)  # mayúsculas, sin espacios/guiones (conserva "_")

    if _FREQUENCY_PATTERN.search(name):
        return QuantityKind.FREQUENCY, Phase.UNKNOWN, PhasorPart.NONE, None

    is_angle = bool(_ANGLE_SUFFIX.search(name))
    is_mag = bool(_MAG_SUFFIX.search(name))
    if not (is_angle or is_mag):
        return QuantityKind.UNKNOWN, Phase.UNKNOWN, PhasorPart.NONE, None
    part = PhasorPart.ANGLE if is_angle else PhasorPart.MAGNITUDE

    # 1) ¿Secuencia (I_0/I_1/I_2/V_0/V_1/V_2)?
    for (domain_letter, kind), pattern in _SEQUENCE_TOKEN_PATTERNS.items():
        if pattern.search(name):
            domain = "current" if domain_letter == "I" else "voltage"
            return kind, Phase.UNKNOWN, part, domain  # la secuencia no es de una fase específica

    # 2) ¿Magnitud/ángulo fasorial por fase (Ia Mag, Vb Angle, etc.)?
    for phase, pattern in _PHASOR_PHASE_PATTERNS.items():
        match = pattern.search(name)
        if match:
            # el patrón exige que el caracter I o V preceda inmediatamente
            # a la letra de fase (ver _PHASOR_PHASE_PATTERNS); se recupera
            # revisando el propio match.
            domain = "current" if match.group(0)[0] == "I" else "voltage"
            return QuantityKind.PHASOR_PHASE, phase, part, domain

    # Tiene sufijo Mag/Angle pero no calzó ningún patrón conocido —
    # se reporta como fasor no identificado, no como forma de onda.
    return QuantityKind.UNKNOWN, Phase.UNKNOWN, part, None


def classify_phase_by_ph_field(ph_field: str) -> Phase | None:
    """
    Usa el campo `ph` del CFG cuando el fabricante lo llena según el
    estándar (IEEE C37.111 sugiere valores como 'A', 'B', 'C', 'N').
    Retorna None si el campo está vacío o no es interpretable.
    """
    p = ph_field.strip().upper()
    if not p:
        return None
    mapping = {
        "A": Phase.A, "AN": Phase.A, "AG": Phase.A, "1": Phase.A,
        "B": Phase.B, "BN": Phase.B, "BG": Phase.B, "2": Phase.B,
        "C": Phase.C, "CN": Phase.C, "CG": Phase.C, "3": Phase.C,
        "N": Phase.NEUTRAL, "G": Phase.NEUTRAL, "E": Phase.NEUTRAL, "0": Phase.NEUTRAL,
    }
    return mapping.get(p)


# ---------------------------------------------------------------------------
# Identificador principal
# ---------------------------------------------------------------------------

@dataclass
class ChannelIdentifier:
    cfg_path: str
    dat_path: str | None = None
    rec: comtrade.Comtrade = field(init=False, default=None)

    def load(self) -> "ChannelIdentifier":
        self.rec = comtrade.Comtrade()
        if self.dat_path:
            self.rec.load(self.cfg_path, self.dat_path)
        else:
            self.rec.load(self.cfg_path)
        return self

    def identify_channels(self) -> list[ChannelInfo]:
        if self.rec is None:
            self.load()

        results: list[ChannelInfo] = []
        for i, chn in enumerate(self.rec.cfg.analog_channels):
            signal_type = classify_by_unit(chn.uu)

            # 0) primero se revisa si es una cantidad ya precalculada por
            #    el IED (secuencia, magnitud fasorial por fase, frecuencia)
            #    — esto tiene prioridad porque, de no chequearse antes,
            #    un canal como "LINEA Ia Mag" terminaría clasificado como
            #    forma de onda cruda de fase A (falso positivo).
            quantity_kind, precomputed_phase, phasor_part, domain_hint = classify_precomputed_quantity(chn.name)

            if quantity_kind != QuantityKind.UNKNOWN:
                if quantity_kind == QuantityKind.FREQUENCY:
                    phase, connection, confidence = Phase.UNKNOWN, "", "precomputed_frequency"
                elif quantity_kind in (QuantityKind.SEQUENCE_ZERO, QuantityKind.SEQUENCE_POSITIVE, QuantityKind.SEQUENCE_NEGATIVE):
                    phase, connection, confidence = Phase.UNKNOWN, "sequence", "precomputed_sequence"
                else:  # PHASOR_PHASE
                    phase, connection, confidence = precomputed_phase, "phasor", "precomputed_phasor"

                results.append(
                    ChannelInfo(
                        index=i, name=chn.name, unit=chn.uu, signal_type=signal_type,
                        phase=phase, connection=connection, confidence=confidence,
                        quantity_kind=quantity_kind, phasor_part=phasor_part, domain_hint=domain_hint,
                    )
                )
                continue

            # 1) intento con el campo ph del CFG
            phase = classify_phase_by_ph_field(chn.ph)
            connection = ""
            confidence = "ph_field" if phase else ""

            # 2) si no hay ph útil, se infiere del nombre del canal
            if phase is None:
                phase, connection = classify_phase_by_name(chn.name)
                if phase != Phase.UNKNOWN:
                    confidence = "name_pattern"
                else:
                    # 2.5) fallback de menor confianza: nombre con prefijo
                    # de estación/línea antepuesto (ver classify_phase_by_suffix)
                    phase, connection = classify_phase_by_suffix(chn.name)
                    confidence = "name_suffix_pattern" if phase != Phase.UNKNOWN else "unclassified"

            # 3) si la unidad tampoco fue concluyente, se intenta también
            #    inferir el tipo de señal desde el nombre (p. ej. si uu
            #    viene vacío pero el canal se llama "IA")
            if signal_type == SignalType.UNKNOWN:
                upper_name = _normalize(chn.name)
                if upper_name.startswith("I") or upper_name.startswith("3I"):
                    signal_type = SignalType.CURRENT
                elif upper_name.startswith(("V", "U")) or upper_name.startswith("3V") or upper_name.startswith("3U"):
                    signal_type = SignalType.VOLTAGE

            results.append(
                ChannelInfo(
                    index=i,
                    name=chn.name,
                    unit=chn.uu,
                    signal_type=signal_type,
                    phase=phase,
                    connection=connection,
                    confidence=confidence,
                    quantity_kind=QuantityKind.WAVEFORM,
                    phasor_part=PhasorPart.NONE,
                )
            )
        return results

    def get_mapping(self) -> dict:
        """
        Agrupa el resultado en un diccionario listo para usar por el
        extractor de features:
            {
              "waveform": {"current": {"a": [idx,...], ...}, "voltage": {...}},
              "precomputed_sequence": {"current": {"zero": idx, "positive": idx, ...}, "voltage": {...}},
              "precomputed_phasor_phase": {"current": {"a": {"magnitude": idx, "angle": idx}, ...}, "voltage": {...}},
              "frequency": idx | None,
              "unclassified": [...],
            }

        La separación entre "waveform" (formas de onda crudas, para el
        extractor de RMS/DFT) y "precomputed_*" (valores que el propio
        IED ya calculó) es intencional: mezclar ambos en el mismo bucket
        llevaría a que el extractor de features intente tratar una
        magnitud fasorial ya calculada como si fuera una muestra cruda
        de corriente/tensión.

        Si hay múltiples canales candidatos para la misma fase de forma
        de onda (p. ej. corrientes de dos devanados), se listan todos
        para que el siguiente paso del pipeline decida cuál usar.
        """
        mapping = {
            "waveform": {"current": {}, "voltage": {}},
            "precomputed_sequence": {"current": {}, "voltage": {}},
            "precomputed_phasor_phase": {"current": {}, "voltage": {}},
            "frequency": None,
            "unclassified": [],
        }

        seq_key_by_kind = {
            QuantityKind.SEQUENCE_ZERO: "zero",
            QuantityKind.SEQUENCE_POSITIVE: "positive",
            QuantityKind.SEQUENCE_NEGATIVE: "negative",
        }

        for ch in self.identify_channels():
            # Para formas de onda crudas, el dominio (corriente/tensión) sale
            # de la unidad eléctrica (uu). Para cantidades precalculadas
            # (secuencia, fasor por fase), se usa domain_hint, porque los
            # canales de ángulo vienen en "deg" y su `uu` no sirve para esto.
            if ch.quantity_kind == QuantityKind.WAVEFORM:
                signal_bucket_name = (
                    "current" if ch.signal_type == SignalType.CURRENT
                    else "voltage" if ch.signal_type == SignalType.VOLTAGE
                    else None
                )
            else:
                signal_bucket_name = ch.domain_hint

            if ch.quantity_kind == QuantityKind.FREQUENCY:
                mapping["frequency"] = ch.index
                continue

            if ch.quantity_kind in seq_key_by_kind:
                if signal_bucket_name is None:
                    mapping["unclassified"].append({"index": ch.index, "name": ch.name, "unit": ch.unit})
                    continue
                seq_key = seq_key_by_kind[ch.quantity_kind]
                bucket = mapping["precomputed_sequence"][signal_bucket_name].setdefault(seq_key, {})
                bucket[ch.phasor_part.value] = ch.index
                continue

            if ch.quantity_kind == QuantityKind.PHASOR_PHASE:
                if signal_bucket_name is None or ch.phase == Phase.UNKNOWN:
                    mapping["unclassified"].append({"index": ch.index, "name": ch.name, "unit": ch.unit})
                    continue
                phase_key = ch.phase.value.lower()
                bucket = mapping["precomputed_phasor_phase"][signal_bucket_name].setdefault(phase_key, {})
                bucket[ch.phasor_part.value] = ch.index
                continue

            # Forma de onda cruda (comportamiento original)
            if signal_bucket_name is None or ch.phase == Phase.UNKNOWN:
                mapping["unclassified"].append({"index": ch.index, "name": ch.name, "unit": ch.unit})
                continue

            key = ch.phase.value.lower()
            mapping["waveform"][signal_bucket_name].setdefault(key, []).append(ch.index)

        return mapping

    def report(self) -> str:
        """Resumen legible para revisión manual / logs."""
        header = (
            f"{'#':>3}  {'Nombre':<18}  {'Unidad':<7}  {'Tipo':<9}  {'Fase':<8}  "
            f"{'Cantidad':<19}  {'Parte':<9}  {'Origen'}"
        )
        lines = [header]
        for ch in self.identify_channels():
            lines.append(
                f"{ch.index:>3}  {ch.name:<18}  {ch.unit:<7}  {ch.signal_type.value:<9}  {ch.phase.value:<8}  "
                f"{ch.quantity_kind.value:<19}  {ch.phasor_part.value:<9}  {ch.confidence}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Uso de ejemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    identifier = ChannelIdentifier(cfg_path=r"comtrades\sample7\oscilografia.cfg").load()
    print(identifier.report())
    print()
    import json
    print(json.dumps(identifier.get_mapping(), indent=2, default=str))
