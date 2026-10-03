from __future__ import annotations

import numpy as np


def scli_metrics(labels: np.ndarray, records: list[dict]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    raw = np.asarray([r["raw_label"] for r in records], dtype=int)
    known = np.asarray([r["known"] for r in records], dtype=bool)
    pred = np.asarray([(-1 if r["label"] is None else r["label"]) for r in records], dtype=int)

    positive = y == 1
    negative = y == 0
    false_clear = float(np.mean(known[positive] & (pred[positive] == 0))) if positive.any() else np.nan
    false_alarm = float(np.mean(known[negative] & (pred[negative] == 1))) if negative.any() else np.nan
    known_accuracy = float(np.mean(pred[known] == y[known])) if known.any() else np.nan

    return {
        "scli_raw_accuracy": float(np.mean(raw == y)),
        "known_coverage": float(known.mean()),
        "known_accuracy": known_accuracy,
        "underdetermined_rate": float(1.0 - known.mean()),
        "false_clear_rate": false_clear,
        "false_alarm_rate": false_alarm,
    }
