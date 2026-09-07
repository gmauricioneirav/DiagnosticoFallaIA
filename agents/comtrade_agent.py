"""
agents/comtrade_agent.py

Nodo de Ingesta: parseo puro del archivo COMTRADE (sin LLM, sin cálculo
de features -- eso lo hace agents/feature_agent.py vía herramientas MCP).
Usa ComtradeFeatureExtractor solo para cargar el registro y leer
metadatos -- el objeto `comtrade.Comtrade` real que expone, con los
atributos que sí tiene: station_name, rec_dev_id, cfg.sample_rates,
cfg.frequency, time, trigger_time, analog_channel_ids,
status_channel_ids.
"""

from __future__ import annotations

from graph.workflow import FaultAnalysisState
from tools.comtrade_features import ComtradeFeatureExtractor


def ingesta_node(state: FaultAnalysisState) -> dict:
    cfg_path, dat_path = state.get("cfg_path"), state.get("dat_path")
    if not cfg_path or not dat_path:
        raise ValueError("ingesta_node requiere 'cfg_path' y 'dat_path' en el State.")

    extractor = ComtradeFeatureExtractor(cfg_path=cfg_path, dat_path=dat_path).load()
    rec = extractor.rec

    raw_metadata = {
        "station_name": rec.station_name,
        "device_id": rec.rec_dev_id,
        "sample_rate": rec.cfg.sample_rates[0][0] if rec.cfg.sample_rates else None,
        "nominal_frequency_hz": rec.cfg.frequency,
        "n_samples": len(rec.time),
        "trigger_time_s": rec.trigger_time,
        "analog_channels": list(rec.analog_channel_ids),
        "digital_channels": list(rec.status_channel_ids),
    }
    return {"raw_metadata": raw_metadata}
