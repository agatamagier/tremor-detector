"""
build_td_rest.py
================
Buduje zbiór okien do eksperymentu docelowego: **PD tremor-dominant (TD-PD)
vs zdrowi**, wyłącznie z **zadań spoczynkowych** (Relaxed + RelaxedTask).

Dlaczego osobny skrypt (a nie preprocess.py + convert):
    - ``convert`` etykietuje tylko po polu diagnozy (0=zdrowy, 1=PD, 2=inne)
      i NIE zna podtypu PD (tremor-dominant vs akinetic/hypokinetic-rigid).
      Podtyp jest w ``disease_comment`` pacjenta.
    - Filtr podtypu + filtr zadania są kluczem eksperymentu 7 (patrz
      DZIENNIK_BADAWCZY). Ten builder robi oba w jednym, odtwarzalnym miejscu,
      czytając surowe .txt bezpośrednio (bez pośredniego CSV).

Wejście:
    - lista ID kohorty: ``cohort_td_healthy.json`` = {"td_pd":[...], "healthy":[...]}
      (zbudowana z pełnego parsowania disease_comment wszystkich ~469 etykiet).
    - surowe sygnały: ``data/raw/pads_dataset/movement/timeseries/*.txt``.

Przetwarzanie 1 pliku = resample(fs=100) -> bandpass 3-12 Hz (filtfilt) ->
okna 3 s / 50% overlap (identycznie jak preprocess.py).

Etykiety:   zdrowy -> 0,  TD-PD -> 1.
Zadania:    tylko spoczynkowe (nazwa zawiera "relaxed"); StretchHold i kinetyczne
            pomijane.

Wyjście: ``data/processed/windows_td_rest.npz`` z kluczami X/y/subjects/sources/
         channels/fs/window_sec/overlap (zgodnie z train_freq.py / eval).

Użycie:
    python build_td_rest.py --cohort cohort_td_healthy.json \
        --root data/raw/pads_dataset --out data/processed/windows_td_rest.npz
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd

from preprocess import (
    PreprocessConfig,
    SENSOR_CHANNELS,
    apply_bandpass,
    resample_uniform,
    sliding_windows,
)
from download_pads import (
    _load_signal,
    _parse_signal_name,
    _resolve_layout,
    _split_time_channels,
)


def _is_rest_task(task: str) -> bool:
    """Zadanie spoczynkowe = nazwa zawiera 'relaxed' (Relaxed / RelaxedTask).

    StretchHold (posturalne) i zadania kinetyczne są odrzucane — drżenie
    spoczynkowe PD (marker 4-6 Hz) jest w spoczynku najczystsze.
    """
    return "relaxed" in task.lower()


def _signal_to_dataframe(signal_array: np.ndarray) -> pd.DataFrame | None:
    """Surowa macierz .txt -> DataFrame [timestamp + 6 kanałów] dla preprocess."""
    split = _split_time_channels(signal_array)
    if split is None:
        return None
    timestamps, channel_data = split
    frame = {"timestamp": timestamps}
    for channel_idx, channel in enumerate(SENSOR_CHANNELS):
        frame[channel] = channel_data[:, channel_idx]
    return pd.DataFrame(frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Builder zbioru TD-PD vs zdrowi (spoczynek) dla train_freq.py.")
    parser.add_argument("--cohort", default="cohort_td_healthy.json")
    parser.add_argument("--root", default="data/raw/pads_dataset")
    parser.add_argument("--out", default="data/processed/windows_td_rest.npz")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.cohort, "r", encoding="utf-8") as cohort_file:
        cohort = json.load(cohort_file)
    # id -> etykieta (0 zdrowy / 1 TD-PD). ID normalizujemy do str bez zer wiodących
    # (klucz dopasowania do subject z nazwy pliku, który też normalizujemy).
    label_by_subject: dict[str, int] = {}
    for raw_id in cohort["healthy"]:
        label_by_subject[str(int(raw_id))] = 0
    for raw_id in cohort["td_pd"]:
        label_by_subject[str(int(raw_id))] = 1
    print(f"[build] Kohorta: {sum(v == 1 for v in label_by_subject.values())} TD-PD, "
          f"{sum(v == 0 for v in label_by_subject.values())} zdrowych "
          f"(razem {len(label_by_subject)} pacjentów).")

    layout = _resolve_layout(args.root)
    signal_paths = sorted(glob(os.path.join(layout["timeseries"], "*.txt")))
    print(f"[build] Plików .txt w timeseries: {len(signal_paths)}")

    config = PreprocessConfig(fs=args.fs, lowcut=3.0, highcut=12.0, order=4,
                              window_sec=args.window_sec, overlap=args.overlap)

    all_windows: list[np.ndarray] = []
    sources: list[str] = []
    labels: list[int] = []
    subjects: list[str] = []
    used_subjects: set[str] = set()
    n_skip_task = n_skip_notcohort = n_skip_shape = n_skip_short = 0

    for path in signal_paths:
        info = _parse_signal_name(path)
        if info is None:
            continue
        subject = str(int(info["subject"]))  # normalizacja: '005' -> '5'
        if subject not in label_by_subject:
            n_skip_notcohort += 1
            continue
        if not _is_rest_task(info["task"]):
            n_skip_task += 1
            continue

        signal_array = _load_signal(path)
        signal_df = (_signal_to_dataframe(signal_array)
                     if signal_array is not None else None)
        if signal_df is None:
            n_skip_shape += 1
            continue
        try:
            uniform = resample_uniform(signal_df, config.fs, config.channels)
            filtered = apply_bandpass(uniform, config)
            windows = sliding_windows(filtered, config)
        except ValueError:
            n_skip_short += 1
            continue
        if windows.shape[0] == 0:
            n_skip_short += 1
            continue

        label = label_by_subject[subject]
        all_windows.append(windows)
        sources.extend([os.path.basename(path)] * windows.shape[0])
        labels.extend([label] * windows.shape[0])
        subjects.extend([subject] * windows.shape[0])
        used_subjects.add(subject)

    if not all_windows:
        raise SystemExit("[build] Brak okien — sprawdź, czy sygnały kohorty są pobrane.")

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    subjects_arr = np.array(subjects)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        subjects=subjects_arr,
        sources=np.array(sources),
        channels=np.array(config.channels),
        fs=config.fs,
        window_sec=config.window_sec,
        overlap=config.overlap,
    )

    n_healthy_pat = sum(1 for s in used_subjects if label_by_subject[s] == 0)
    n_td_pat = sum(1 for s in used_subjects if label_by_subject[s] == 1)
    print(f"\n[build] Zapisano {X.shape[0]} okien, kształt {X.shape} -> {args.out}")
    print(f"[build] Okna wg klasy: 0 (zdrowy)={int((y == 0).sum())}, "
          f"1 (TD-PD)={int((y == 1).sum())}")
    print(f"[build] Pacjentów z danymi: {len(used_subjects)} "
          f"({n_healthy_pat} zdrowych, {n_td_pat} TD-PD).")
    print(f"[build] Pominięto: nie-kohorta={n_skip_notcohort}, nie-spoczynek={n_skip_task}, "
          f"zły kształt={n_skip_shape}, za krótkie={n_skip_short}.")
    missing = sorted(set(label_by_subject) - used_subjects, key=int)
    if missing:
        print(f"[build] UWAGA: {len(missing)} pacjentów kohorty bez danych spoczynkowych "
              f"na dysku (do dociągnięcia): {missing[:30]}{' ...' if len(missing) > 30 else ''}")


if __name__ == "__main__":
    main()
