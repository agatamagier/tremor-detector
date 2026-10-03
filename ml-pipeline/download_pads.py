"""
download_pads.py
================
Pobranie i adaptacja zbioru PADS (Parkinson's Disease Smartwatch) do formatu
oczekiwanego przez `preprocess.py`.

Zbiór PADS (open access) jest hostowany na PhysioNet:
    https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/

Struktura surowych danych PADS (po rozpakowaniu):
    <root>/
    ├── patients/        *.json  — metadane pacjenta (subject_id, condition = ETYKIETA)
    ├── movement/        *.json (meta) + *.txt (sygnały, np.loadtxt, delimiter=",")
    └── questionnaire/   *.json  — 30 pytań PDNMS (nieużywane w detekcji ruchu)

Co robi ten skrypt (3 tryby):
    download  — pobiera i rozpakowuje zbiór z PhysioNet.
    inspect   — wypisuje wykryte pola/kanały dla PIERWSZEGO pacjenta, żeby
                zweryfikować założenia na realnych plikach PRZED masową konwersją.
    convert   — konwertuje sygnały ruchowe na nasz schemat CSV:
                kolumny [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z],
                jeden plik na (subject, task, wrist), + labels.csv z etykietą diagnozy.

Mapowanie etykiet (jak w oryginalnym repo PADS, cel = detekcja PD):
    Healthy                     -> 0
    Parkinson's                 -> 1
    Essential Tremor / MS /
    Atypical / Other disorders  -> 2

UWAGA o częstotliwości próbkowania:
    Smartwatch PADS próbkuje ~100 Hz (repo odrzuca "pierwsze 0.5 s = 48 próbek").
    Dlatego po konwersji uruchamiaj preprocess.py z --fs 100 (nie 50).

Użycie:
    python download_pads.py download --root data/raw/pads_dataset
    python download_pads.py inspect  --root data/raw/pads_dataset
    python download_pads.py convert  --root data/raw/pads_dataset --out data/raw/pads_csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from glob import glob
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

PHYSIONET_ZIP = (
    "https://physionet.org/static/published-projects/"
    "parkinsons-disease-smartwatch/"
    "parkinsons-disease-smartwatch-1.0.0.zip"
)

#: Zakładana częstotliwość próbkowania [Hz] (patrz uwaga w nagłówku).
PADS_FS = 100.0

#: Mapowanie diagnozy -> klasa (detekcja PD).
CONDITION_TO_LABEL = {
    "Healthy": 0,
    "Parkinson's": 1,
    "Essential Tremor": 2,
    "Multiple Sclerosis": 2,
    "Atypical Parkinsonism": 2,
    "Other Movement Disorders": 2,
}

#: Nasz docelowy, stały schemat kanałów (kolejność jak w preprocess.py).
TARGET_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


# ---------------------------------------------------------------------------
# Flatten zagnieżdżonych metadanych (port z pads-project/preprocessing)
# ---------------------------------------------------------------------------

def _flatten_dict(data_dict, tmp_fields, data_dict_flat):
    is_most_inner = True
    for key, item in data_dict.items():
        if isinstance(item, dict):
            is_most_inner = False
            _flatten_dict(item, tmp_fields.copy(), data_dict_flat)
        elif isinstance(item, list) and item and isinstance(item[0], dict):
            is_most_inner = False
            for list_item in item:
                _flatten_dict(list_item, tmp_fields.copy(), data_dict_flat)
        else:
            tmp_fields[key] = item
    if is_most_inner:
        data_dict_flat.append(tmp_fields)
    return data_dict_flat


def flatten_json(path: str) -> pd.DataFrame:
    """Wczytuje jeden plik .json i spłaszcza do DataFrame (jak w repo PADS)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = _flatten_dict(data, {}, [])
    return pd.DataFrame(rows)


def load_meta_dir(path: str) -> list[pd.DataFrame]:
    """Wczytuje wszystkie *.json z katalogu, posortowane."""
    files = sorted(glob(os.path.join(path, "*.json")))
    return [flatten_json(f) for f in files]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def cmd_download(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "pads.zip")

    if not os.path.exists(zip_path):
        print(f"[download] Pobieram PADS z PhysioNet...\n  {PHYSIONET_ZIP}")
        print("  (to kilkaset MB — może chwilę potrwać)")
        urlretrieve(PHYSIONET_ZIP, zip_path)
    else:
        print(f"[download] Archiwum już istnieje: {zip_path}")

    print("[download] Rozpakowuję...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    print(f"[download] Gotowe. Sprawdź strukturę pod: {root}")
    print("[download] Następnie: python download_pads.py inspect --root <ten_katalog>")


def _resolve_subdirs(root: str) -> dict[str, str]:
    """
    Znajduje podkatalogi patients/ movement/ questionnaire/ — mogą być
    zagnieżdżone o jeden poziom (nazwa wersji) po rozpakowaniu zipa.
    """
    for base in (root, *glob(os.path.join(root, "*"))):
        if not os.path.isdir(base):
            continue
        movement = os.path.join(base, "movement")
        patients = os.path.join(base, "patients")
        if os.path.isdir(movement) and os.path.isdir(patients):
            return {
                "base": base,
                "patients": patients + os.sep,
                "movement": movement + os.sep,
                "questionnaire": os.path.join(base, "questionnaire") + os.sep,
            }
    raise FileNotFoundError(
        f"Nie znaleziono patients/ i movement/ pod {root}. "
        f"Czy zbiór został pobrany i rozpakowany (tryb 'download')?"
    )


# ---------------------------------------------------------------------------
# Inspect — weryfikacja realnej struktury przed konwersją
# ---------------------------------------------------------------------------

def cmd_inspect(root: str) -> None:
    dirs = _resolve_subdirs(root)
    print(f"[inspect] Katalog bazowy: {dirs['base']}")

    patients = load_meta_dir(dirs["patients"])
    print(f"[inspect] Liczba plików pacjentów: {len(patients)}")
    if patients:
        p = patients[0]
        print(f"[inspect] Kolumny meta pacjenta: {list(p.columns)}")
        for col in ("subject_id", "condition", "label"):
            if col in p.columns:
                print(f"[inspect]   {col}: {p[col].iloc[0]}")

    movement = load_meta_dir(dirs["movement"])
    print(f"[inspect] Liczba plików ruchu: {len(movement)}")
    if movement:
        m = movement[0]
        print(f"[inspect] Kolumny meta ruchu: {list(m.columns)}")
        # Wypisz kilka pierwszych rekordów (nazwy plików txt + kanały).
        cols = [c for c in ("record_name", "device_location", "channels",
                            "file_name", "rows", "subject_id") if c in m.columns]
        print(f"[inspect] Przykładowe rekordy:\n{m[cols].head(8).to_string()}")

        # Spróbuj wczytać pierwszy sygnał txt i podaj kształt.
        if "file_name" in m.columns:
            txt = os.path.join(dirs["movement"], m["file_name"].iloc[0])
            if os.path.exists(txt):
                arr = np.loadtxt(txt, dtype=np.float32, delimiter=",")
                print(f"[inspect] Pierwszy sygnał {m['file_name'].iloc[0]}: "
                      f"shape={arr.shape} (wiersze=próbki, kolumny=kanały)")


# ---------------------------------------------------------------------------
# Convert — PADS -> nasz CSV
# ---------------------------------------------------------------------------

def _axis_from_channel(name: str) -> str | None:
    """Zwraca 'accel_x'.. 'gyro_z' na podstawie nazwy kanału PADS, albo None."""
    low = name.lower()
    if "acc" in low:
        sensor = "accel"
    elif "gyro" in low or "rot" in low:
        sensor = "gyro"
    else:
        return None
    m = re.search(r"(?:^|_)([xyz])(?:$|_)", low)
    if not m:
        return None
    return f"{sensor}_{m.group(1)}"


def _patient_labels(patients: list[pd.DataFrame]) -> dict[str, dict]:
    """subject_id -> {condition, label} na podstawie metadanych pacjentów."""
    out: dict[str, dict] = {}
    for p in patients:
        if "subject_id" not in p.columns or "condition" not in p.columns:
            continue
        sid = str(p["subject_id"].iloc[0])
        condition = str(p["condition"].iloc[0])
        label = CONDITION_TO_LABEL.get(condition, 2)
        out[sid] = {"condition": condition, "label": label}
    return out


def cmd_convert(root: str, out_dir: str, min_rows: int = 100) -> None:
    """
    Konwertuje sygnały ruchowe PADS na pliki CSV w naszym schemacie.

    Grupuje kanały po (record_name/zadanie, device_location/nadgarstek) i składa
    dla każdej grupy ramkę [timestamp, accel_x..gyro_z]. Grupy, które nie mają
    kompletu 6 kanałów (np. tylko akcelerometr), są pomijane z ostrzeżeniem.
    """
    dirs = _resolve_subdirs(root)
    os.makedirs(out_dir, exist_ok=True)

    patients = load_meta_dir(dirs["patients"])
    labels = _patient_labels(patients)
    print(f"[convert] Wczytano etykiety dla {len(labels)} pacjentów.")

    movement = load_meta_dir(dirs["movement"])
    label_rows: list[dict] = []
    n_written = 0

    for meta in movement:
        if meta.empty or "file_name" not in meta.columns:
            continue
        sid = str(meta["subject_id"].iloc[0]) if "subject_id" in meta.columns else "unknown"
        label_info = labels.get(sid, {"condition": "unknown", "label": -1})

        # Grupy sygnałów: (zadanie, nadgarstek) -> { 'accel_x': kolumna, ... }
        groups: dict[tuple, dict[str, np.ndarray]] = {}

        for _, rec in meta.iterrows():
            txt = os.path.join(dirs["movement"], rec["file_name"])
            if not os.path.exists(txt):
                continue
            arr = np.loadtxt(txt, dtype=np.float32, delimiter=",")
            if arr.ndim == 1:
                arr = arr[:, None]

            channels = list(rec.get("channels", []))
            if len(channels) != arr.shape[1]:
                # Dopasowanie kolumn do nazw kanałów nie wychodzi — pomiń rekord.
                print(f"[convert] POMINIĘTO rekord {rec.get('file_name')}: "
                      f"{len(channels)} nazw kanałów != {arr.shape[1]} kolumn.")
                continue

            task = str(rec.get("record_name", "task"))
            wrist = str(rec.get("device_location", "wrist"))
            key = (task, wrist)
            grp = groups.setdefault(key, {})

            for col_idx, ch_name in enumerate(channels):
                target = _axis_from_channel(str(ch_name))
                if target is not None:
                    grp[target] = arr[:, col_idx]

        # Zapisz kompletne grupy jako CSV.
        for (task, wrist), grp in groups.items():
            if not all(ch in grp for ch in TARGET_CHANNELS):
                continue  # brak kompletu accel+gyro xyz
            length = min(len(grp[ch]) for ch in TARGET_CHANNELS)
            if length < min_rows:
                continue
            t = np.arange(length) / PADS_FS
            frame = {"timestamp": t}
            for ch in TARGET_CHANNELS:
                frame[ch] = grp[ch][:length]
            df = pd.DataFrame(frame)

            fname = f"{sid}_{task}_{wrist}.csv".replace(os.sep, "_")
            df.to_csv(os.path.join(out_dir, fname), index=False)
            label_rows.append({
                "filename": fname,
                "subject_id": sid,
                "condition": label_info["condition"],
                "label": label_info["label"],
            })
            n_written += 1

    labels_csv = os.path.join(out_dir, "labels.csv")
    pd.DataFrame(label_rows).to_csv(labels_csv, index=False)
    print(f"[convert] Zapisano {n_written} plików CSV -> {out_dir}")
    print(f"[convert] Etykiety -> {labels_csv}")
    print(f"[convert] Dalej: python preprocess.py --input {out_dir} "
          f"--output data/processed --fs {int(PADS_FS)} --labels {labels_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Pobieranie i adaptacja zbioru PADS.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Pobierz i rozpakuj PADS z PhysioNet.")
    d.add_argument("--root", default="data/raw/pads_dataset")

    i = sub.add_parser("inspect", help="Wypisz strukturę/kanały pierwszego pacjenta.")
    i.add_argument("--root", default="data/raw/pads_dataset")

    c = sub.add_parser("convert", help="Konwertuj PADS -> nasz schemat CSV.")
    c.add_argument("--root", default="data/raw/pads_dataset")
    c.add_argument("--out", default="data/raw/pads_csv")
    c.add_argument("--min-rows", type=int, default=100)

    args = p.parse_args()
    if args.cmd == "download":
        cmd_download(args.root)
    elif args.cmd == "inspect":
        cmd_inspect(args.root)
    elif args.cmd == "convert":
        cmd_convert(args.root, args.out, args.min_rows)


if __name__ == "__main__":
    main()
