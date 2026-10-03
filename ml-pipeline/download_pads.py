"""
download_pads.py
================
Adaptacja zbioru PADS (Parkinson's Disease Smartwatch) do formatu oczekiwanego
przez ``preprocess.py``.

Zbiór PADS (open access) hostowany na PhysioNet:
    https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/

REALNA struktura danych (pliki można pobrać ręcznie, bez całego ZIP-a 770 MB):

    <root>/
    ├── movement/
    │   ├── observation_001.json      # metadane obserwacji (opcjonalne u nas)
    │   ├── observation_002.json
    │   └── timeseries/
    │       ├── 001_Relaxed_LeftWrist.txt      # sygnał: subject_zadanie_nadgarstek
    │       ├── 001_Relaxed_RightWrist.txt
    │       ├── 001_RelaxedTask_LeftWrist.txt
    │       ├── 001_StretchHold_LeftWrist.txt
    │       └── ...
    └── patients/
        ├── patient_001.json          # metadane pacjenta (ETYKIETA: condition/diagnosis)
        └── patient_002.json

KLUCZOWE: zadanie i nadgarstek są zakodowane w NAZWIE pliku .txt
(``<subject>_<task>_<wrist>.txt``), a nie w metadanych JSON.

Do DETEKCJI DRŻENIA bierzemy tylko zadania spoczynkowe/posturalne
(domyślny filtr ``TREMOR_TASKS``):
    - Relaxed       — drżenie spoczynkowe (klasyczne PD),
    - RelaxedTask   — spoczynek + zadanie poznawcze,
    - StretchHold   — drżenie posturalne (ET / PD).
Zadania celowe/kinetyczne (DrinkGlas, CrossArms, PointFinger...) pomijamy.

Tryby:
    download  — (opcjonalnie) pobiera CAŁY ZIP z PhysioNet i rozpakowuje.
                W praktyce pliki pobiera się ręcznie — patrz struktura wyżej.
    inspect   — wypisuje realny format (klucze JSON pacjenta, wykryte zadania,
                kształt pierwszego sygnału). OBOWIĄZKOWO przed convert.
    convert   — konwertuje wybrane zadania na nasz schemat CSV:
                [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                + labels.csv z etykietą diagnozy.

Mapowanie etykiet (cel = detekcja PD):
    Healthy / Control           -> 0
    Parkinson's                 -> 1
    inne (ET / MS / Atypical...) -> 2

UWAGA o fs: smartwatch PADS próbkuje ~100 Hz -> preprocess.py z --fs 100.

Użycie:
    python download_pads.py inspect --root data/raw
    python download_pads.py convert --root data/raw --out data/raw/pads_csv
    python download_pads.py convert --root data/raw --out data/raw/pads_csv --tasks Relaxed StretchHold
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

#: Baza URL do pojedynczych plików projektu na PhysioNet (tryb `fetch`).
PHYSIONET_FILES_BASE = (
    "https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
)

#: Manifest wszystkich plików projektu (lista ścieżek) — używany przez `fetch`,
#: żeby nie zgadywać dokładnych nazw zadań.
PHYSIONET_MANIFEST = PHYSIONET_FILES_BASE + "SHA256SUMS.txt"

#: Zakładana częstotliwość próbkowania [Hz] (patrz uwaga w nagłówku).
PADS_FS = 100.0

#: Domyślny filtr zadań istotnych dla DRŻENIA (dopasowanie: podłańcuch, bez
#: rozróżniania wielkości liter). "Relaxed" złapie też "RelaxedTask"/"Relaxed1".
TREMOR_TASKS = ["Relaxed", "StretchHold"]

#: Nasz docelowy, stały schemat kanałów (kolejność jak w preprocess.py).
#: ZAŁOŻENIE kolejności kolumn w .txt: 3x akcelerometr, potem 3x żyroskop.
#: Zweryfikuj trybem `inspect` (wypisuje kształt i pierwsze wiersze).
TARGET_CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]

#: Wzorzec nazwy pliku sygnału: <subject>_<task>_<wrist>.txt
FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<task>.+)_(?P<wrist>[A-Za-z]+Wrist)\.txt$"
)


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


# ---------------------------------------------------------------------------
# Rozpoznanie struktury katalogów
# ---------------------------------------------------------------------------

def _resolve_layout(root: str) -> dict[str, str]:
    """
    Znajduje katalogi movement/ i patients/ (mogą być zagnieżdżone o jeden
    poziom po rozpakowaniu ZIP-a). Sygnały .txt są w movement/timeseries/
    (z fallbackiem na movement/ bezpośrednio).
    """
    for base in (root, *sorted(glob(os.path.join(root, "*")))):
        if not os.path.isdir(base):
            continue
        movement = os.path.join(base, "movement")
        patients = os.path.join(base, "patients")
        if os.path.isdir(movement) and os.path.isdir(patients):
            ts = os.path.join(movement, "timeseries")
            timeseries = ts if os.path.isdir(ts) else movement
            return {
                "base": base,
                "patients": patients,
                "movement": movement,
                "timeseries": timeseries,
            }
    raise FileNotFoundError(
        f"Nie znaleziono movement/ i patients/ pod {root}. "
        f"Sprawdź strukturę (patrz nagłówek download_pads.py)."
    )


def _list_signal_files(timeseries_dir: str) -> list[str]:
    """Lista plików .txt z sygnałami, posortowana."""
    return sorted(glob(os.path.join(timeseries_dir, "*.txt")))


def _parse_signal_name(path: str) -> dict | None:
    """Z nazwy pliku wyciąga {subject, task, wrist} albo None, gdy nie pasuje."""
    m = FILENAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return {"subject": m.group("subject"),
            "task": m.group("task"),
            "wrist": m.group("wrist")}


def _task_matches(task: str, filters: list[str]) -> bool:
    """Czy nazwa zadania zawiera którykolwiek z filtrów (case-insensitive)?"""
    low = task.lower()
    return any(f.lower() in low for f in filters)


# ---------------------------------------------------------------------------
# Etykiety pacjentów
# ---------------------------------------------------------------------------

def _condition_to_label(condition: str) -> int:
    """
    Mapuje tekst diagnozy na klasę (detekcja PD). Heurystyka odporna na warianty
    zapisu: 0=zdrowy, 1=Parkinson, 2=inne zaburzenie ruchowe.
    """
    low = (condition or "").lower()
    if "healthy" in low or "control" in low:
        return 0
    if "parkinson" in low and "atypical" not in low:
        return 1
    return 2


def _extract_condition(flat: pd.DataFrame) -> str:
    """Szuka w spłaszczonych metadanych pacjenta pola z diagnozą."""
    if flat.empty:
        return "unknown"
    for col in flat.columns:
        if any(key in col.lower() for key in ("condition", "disease", "diagnos")):
            val = flat[col].iloc[0]
            if isinstance(val, str) and val.strip():
                return val
    return "unknown"


def _subject_from_patient_file(path: str) -> str:
    """subject_id z nazwy pliku patient_XXX.json (np. '001')."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def _load_patient_labels(patients_dir: str) -> dict[str, dict]:
    """subject_id -> {condition, label} na podstawie plików patient_*.json."""
    out: dict[str, dict] = {}
    for path in sorted(glob(os.path.join(patients_dir, "*.json"))):
        sid = _subject_from_patient_file(path)
        condition = _extract_condition(flatten_json(path))
        out[sid] = {"condition": condition, "label": _condition_to_label(condition)}
    return out


# ---------------------------------------------------------------------------
# Download (opcjonalny — cały ZIP)
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Pasek postępu dla urlretrieve (ważny przy pobieraniu przez hotspot)."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        bar = "#" * int(pct // 2.5)
        print(f"\r[download] {pct:5.1f}% "
              f"[{bar:<40}] {downloaded/1e6:7.1f} / {total_size/1e6:.1f} MB",
              end="", flush=True)
    else:
        print(f"\r[download] {downloaded/1e6:7.1f} MB", end="", flush=True)


def cmd_download(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "pads.zip")

    if not os.path.exists(zip_path):
        print(f"[download] Pobieram CAŁY PADS z PhysioNet (~770 MB)...\n  {PHYSIONET_ZIP}")
        print("  Wskazówka: zwykle wystarczy pobrać ręcznie tylko wybrane pliki "
              "zadań (Relaxed/RelaxedTask/StretchHold) — patrz nagłówek pliku.")
        urlretrieve(PHYSIONET_ZIP, zip_path, reporthook=_progress_hook)
        print()
    else:
        print(f"[download] Archiwum już istnieje: {zip_path}")

    print("[download] Rozpakowuję...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    print(f"[download] Gotowe. Dalej: python download_pads.py inspect --root {root}")


# ---------------------------------------------------------------------------
# Fetch — celowane pobieranie tylko wybranych zadań/pacjentów
# ---------------------------------------------------------------------------

def _parse_subject_spec(spec: str | None) -> set[int] | None:
    """
    Parsuje specyfikację pacjentów na zbiór numerów (int) albo None = wszyscy.
    Akceptuje np. "1-50", "1,2,3", "1-10,20,30-35".
    """
    if not spec:
        return None
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def _fetch_manifest_paths() -> list[str]:
    """Pobiera SHA256SUMS.txt i zwraca listę ścieżek plików projektu."""
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "pads_SHA256SUMS.txt")
    print(f"[fetch] Pobieram manifest:\n  {PHYSIONET_MANIFEST}")
    urlretrieve(PHYSIONET_MANIFEST, tmp)
    paths: list[str] = []
    with open(tmp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # <hash>  <path>
            if len(parts) != 2:
                continue
            path = parts[1].strip().lstrip("*")  # '*' = marker binarny
            paths.append(path.replace("\\", "/"))
    print(f"[fetch] Manifest: {len(paths)} plików w projekcie.")
    return paths


def _subject_of_path(path: str) -> int | None:
    """Wyciąga numer pacjenta z nazwy pliku (pierwsza grupa cyfr)."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _download_file(rel_path: str, root: str) -> bool:
    """Pobiera pojedynczy plik (skip-if-exists). Zwraca True przy sukcesie."""
    dest = os.path.join(root, rel_path.replace("/", os.sep))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True  # wznawialność — już pobrany
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = PHYSIONET_FILES_BASE + rel_path
    try:
        urlretrieve(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001 — raportujemy i jedziemy dalej
        print(f"\n[fetch] BŁĄD przy {rel_path}: {exc}")
        return False


def cmd_fetch(root: str, tasks: list[str], all_tasks: bool,
              subjects_spec: str | None, limit: int | None) -> None:
    """
    Pobiera z PhysioNet TYLKO wybrane zadania/pacjentów:
      - sygnały movement/timeseries/*.txt pasujące do filtra zadań,
      - patients/patient_XXX.json dla pacjentów, których sygnały pobrano.
    Wymaga internetu bez VPN (np. hotspot). Jest wznawialny (skip-if-exists).
    """
    os.makedirs(root, exist_ok=True)
    subjects = _parse_subject_spec(subjects_spec)

    manifest = _fetch_manifest_paths()

    # Sygnały timeseries pasujące do filtrów (surowe, NIE z preprocessed/).
    ts_paths = [p for p in manifest
                if "timeseries/" in p and p.endswith(".txt")
                and "preprocessed/" not in p]
    selected: list[str] = []
    for p in ts_paths:
        info = _parse_signal_name(p)
        if info is None:
            continue
        if not all_tasks and not _task_matches(info["task"], tasks):
            continue
        subj = _subject_of_path(p)
        if subjects is not None and subj not in subjects:
            continue
        selected.append(p)

    selected.sort()
    if limit is not None:
        selected = selected[:limit]

    subj_present = sorted({_subject_of_path(p) for p in selected if _subject_of_path(p)})
    subj_present_set = set(subj_present)
    patient_paths = [p for p in manifest
                     if "patients/" in p and p.endswith(".json")
                     and "preprocessed/" not in p
                     and _subject_of_path(p) in subj_present_set]

    tasks_present = sorted({_parse_signal_name(p)["task"] for p in selected})
    print(f"[fetch] Filtr zadań: {'(wszystkie)' if all_tasks else tasks}")
    print(f"[fetch] Dopasowane zadania: {tasks_present}")
    print(f"[fetch] Pacjentów: {len(subj_present)}, "
          f"sygnałów do pobrania: {len(selected)}, "
          f"plików pacjentów: {len(patient_paths)}")

    to_download = selected + patient_paths
    ok = 0
    for i, rel in enumerate(to_download, 1):
        print(f"\r[fetch] {i}/{len(to_download)}  {os.path.basename(rel):<40}",
              end="", flush=True)
        if _download_file(rel, root):
            ok += 1
    print(f"\n[fetch] Pobrano {ok}/{len(to_download)} plików -> {root}")
    print(f"[fetch] Dalej: python download_pads.py inspect --root {root}")


# ---------------------------------------------------------------------------
# Inspect — weryfikacja realnej struktury przed konwersją
# ---------------------------------------------------------------------------

def cmd_inspect(root: str, tasks: list[str]) -> None:
    layout = _resolve_layout(root)
    print(f"[inspect] Katalog bazowy:  {layout['base']}")
    print(f"[inspect] patients/:       {layout['patients']}")
    print(f"[inspect] timeseries/:     {layout['timeseries']}")

    # --- Pacjenci / etykiety ---
    patient_files = sorted(glob(os.path.join(layout["patients"], "*.json")))
    print(f"\n[inspect] Plików pacjentów: {len(patient_files)}")
    if patient_files:
        with open(patient_files[0], "r", encoding="utf-8") as f:
            raw = json.load(f)
        print(f"[inspect] Surowe klucze patient_*.json: {list(raw.keys())}")
        flat = flatten_json(patient_files[0])
        print(f"[inspect] Kolumny po spłaszczeniu: {list(flat.columns)}")
        cond = _extract_condition(flat)
        print(f"[inspect] Wykryta diagnoza (condition): '{cond}' "
              f"-> klasa {_condition_to_label(cond)}")

    labels = _load_patient_labels(layout["patients"])
    if labels:
        dist: dict[int, int] = {}
        for info in labels.values():
            dist[info["label"]] = dist.get(info["label"], 0) + 1
        print(f"[inspect] Rozkład etykiet pacjentów: {dict(sorted(dist.items()))}")

    # --- Sygnały ---
    signals = _list_signal_files(layout["timeseries"])
    print(f"\n[inspect] Plików sygnałów (.txt): {len(signals)}")
    parsed = [p for p in (_parse_signal_name(s) for s in signals) if p]
    unmatched = len(signals) - len(parsed)
    if unmatched:
        print(f"[inspect] UWAGA: {unmatched} plików nie pasuje do wzorca "
              f"<subject>_<task>_<wrist>.txt")

    all_tasks = sorted({p["task"] for p in parsed})
    all_wrists = sorted({p["wrist"] for p in parsed})
    print(f"[inspect] Wykryte nadgarstki: {all_wrists}")
    print(f"[inspect] Wykryte zadania ({len(all_tasks)}): {all_tasks}")
    selected = [t for t in all_tasks if _task_matches(t, tasks)]
    print(f"[inspect] Zadania pasujące do filtra {tasks}: {selected}")

    if signals:
        first = signals[0]
        arr = np.loadtxt(first, dtype=np.float32, delimiter=",")
        print(f"\n[inspect] Pierwszy sygnał: {os.path.basename(first)}")
        print(f"[inspect]   shape={arr.shape} (wiersze=próbki, kolumny=kanały)")
        if arr.ndim == 2:
            ncol = arr.shape[1]
            print(f"[inspect]   liczba kolumn: {ncol}")
            if ncol == len(TARGET_CHANNELS) + 1:
                print("[inspect]   -> 7 kolumn = [Time, Accel X/Y/Z, Gyro X/Y/Z] "
                      "(potwierdzone wg pads-project). Kol. 0 (czas) zostanie użyta "
                      "jako timestamp, kol. 1-6 jako kanały.")
            elif ncol == len(TARGET_CHANNELS):
                print("[inspect]   -> 6 kolumn = [Accel X/Y/Z, Gyro X/Y/Z] "
                      "(brak kolumny czasu, timestamp syntetyczny z fs).")
            else:
                print(f"[inspect]   UWAGA: nietypowa liczba kolumn (oczekiwane 6/7).")
            print(f"[inspect]   pierwsze 2 wiersze:\n{arr[:2]}")


# ---------------------------------------------------------------------------
# Convert — PADS -> nasz CSV
# ---------------------------------------------------------------------------

def _load_signal(path: str) -> np.ndarray | None:
    """Wczytuje .txt (delimiter ','); zwraca macierz (próbki, kanały) lub None."""
    arr = np.loadtxt(path, dtype=np.float32, delimiter=",")
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _split_time_channels(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Rozdziela surową macierz PADS na (timestamp, kanały).

    Autorytatywna kolejność kolumn (pads-project/run_preprocessing.py):
        [Time, Accelerometer_X/Y/Z, Gyroscope_X/Y/Z]  -> 7 kolumn.

    Zwraca:
        - 7 kolumn: kolumna 0 jako timestamp [s], kolumny 1..6 jako kanały.
        - 6 kolumn (brak czasu): timestamp syntetyczny z PADS_FS, kolumny 0..5.
        - inaczej: None (zły format).
    """
    n = len(TARGET_CHANNELS)  # 6
    if arr.ndim != 2:
        return None
    if arr.shape[1] == n + 1:          # [Time, accel xyz, gyro xyz]
        return arr[:, 0], arr[:, 1:1 + n]
    if arr.shape[1] == n:              # brak kolumny czasu
        return np.arange(arr.shape[0]) / PADS_FS, arr[:, :n]
    return None


def cmd_convert(root: str, out_dir: str, tasks: list[str],
                all_tasks: bool, min_rows: int = 100) -> None:
    """
    Konwertuje wybrane zadania PADS na pliki CSV w naszym schemacie.

    Jeden plik .txt (jeden subject/task/wrist) -> jeden CSV. Zakładamy 6 kolumn
    w kolejności [accel_x..z, gyro_x..z] (ZWERYFIKOWANE trybem inspect).
    """
    layout = _resolve_layout(root)
    os.makedirs(out_dir, exist_ok=True)

    labels = _load_patient_labels(layout["patients"])
    print(f"[convert] Wczytano etykiety dla {len(labels)} pacjentów.")

    signals = _list_signal_files(layout["timeseries"])
    label_rows: list[dict] = []
    n_written = 0
    n_skipped_task = 0
    n_skipped_shape = 0
    n_skipped_nolabel = 0
    missing_label_subjects: set[str] = set()

    for path in signals:
        info = _parse_signal_name(path)
        if info is None:
            continue
        if not all_tasks and not _task_matches(info["task"], tasks):
            n_skipped_task += 1
            continue

        arr = _load_signal(path)
        split = _split_time_channels(arr) if arr is not None else None
        if split is None:
            print(f"[convert] POMINIĘTO {os.path.basename(path)}: "
                  f"kształt {None if arr is None else arr.shape} "
                  f"(oczekiwane 6 lub 7 kolumn).")
            n_skipped_shape += 1
            continue
        t, data = split
        if data.shape[0] < min_rows:
            n_skipped_shape += 1
            continue

        # data: kolumny w kolejności accel X/Y/Z, gyro X/Y/Z (TARGET_CHANNELS).
        frame = {"timestamp": t}
        for i, ch in enumerate(TARGET_CHANNELS):
            frame[ch] = data[:, i]
        df = pd.DataFrame(frame)

        sid = info["subject"]
        label_info = labels.get(sid, {"condition": "unknown", "label": -1})
        if label_info["label"] < 0:
            # Brak patient_*.json dla tego pacjenta -> nie da się trenować.
            missing_label_subjects.add(sid)
            n_skipped_nolabel += 1
            continue
        fname = f"{sid}_{info['task']}_{info['wrist']}.csv"
        df.to_csv(os.path.join(out_dir, fname), index=False)
        label_rows.append({
            "filename": fname,
            "subject_id": sid,
            "task": info["task"],
            "wrist": info["wrist"],
            "condition": label_info["condition"],
            "label": label_info["label"],
        })
        n_written += 1

    labels_csv = os.path.join(out_dir, "labels.csv")
    pd.DataFrame(label_rows).to_csv(labels_csv, index=False)

    print(f"[convert] Zapisano {n_written} plików CSV -> {out_dir}")
    print(f"[convert] Pominięto (filtr zadań): {n_skipped_task}, "
          f"(zły kształt): {n_skipped_shape}, "
          f"(brak etykiety/patient_*.json): {n_skipped_nolabel}")
    if missing_label_subjects:
        print(f"[convert] UWAGA: brak patient_*.json dla pacjentów "
              f"{sorted(missing_label_subjects)} — dociągnij te pliki, by ich użyć.")
    if label_rows:
        dist: dict[int, int] = {}
        for r in label_rows:
            dist[r["label"]] = dist.get(r["label"], 0) + 1
        print(f"[convert] Rozkład etykiet (okna CSV): {dict(sorted(dist.items()))} "
              f"(0=zdrowy, 1=PD, 2=inne, -1=brak etykiety)")
    print(f"[convert] Etykiety -> {labels_csv}")
    print(f"[convert] Dalej: python preprocess.py --input {out_dir} "
          f"--output data/processed --fs {int(PADS_FS)} --labels {labels_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Adaptacja zbioru PADS -> nasz CSV.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="(Opcjonalnie) pobierz CAŁY ZIP z PhysioNet.")
    d.add_argument("--root", default="data/raw")

    f = sub.add_parser("fetch", help="Pobierz TYLKO wybrane zadania/pacjentów z PhysioNet.")
    f.add_argument("--root", default="data/raw/pads_dataset")
    f.add_argument("--tasks", nargs="+", default=TREMOR_TASKS,
                   help="Filtr zadań (podłańcuchy). Domyślnie: Relaxed, StretchHold.")
    f.add_argument("--all-tasks", action="store_true",
                   help="Pobierz wszystkie zadania (ignoruj filtr).")
    f.add_argument("--subjects", default=None,
                   help="Pacjenci, np. '1-50', '1,2,3', '1-10,20'. Domyślnie: wszyscy.")
    f.add_argument("--limit", type=int, default=None,
                   help="Maks. liczba plików sygnałów do pobrania (test).")

    i = sub.add_parser("inspect", help="Wypisz realną strukturę / zadania / kanały.")
    i.add_argument("--root", default="data/raw")
    i.add_argument("--tasks", nargs="+", default=TREMOR_TASKS,
                   help="Filtr zadań (podłańcuchy). Domyślnie zadania drżenia.")

    c = sub.add_parser("convert", help="Konwertuj wybrane zadania -> nasz CSV.")
    c.add_argument("--root", default="data/raw")
    c.add_argument("--out", default="data/raw/pads_csv")
    c.add_argument("--tasks", nargs="+", default=TREMOR_TASKS,
                   help="Filtr zadań (podłańcuchy). Domyślnie: Relaxed, StretchHold.")
    c.add_argument("--all-tasks", action="store_true",
                   help="Zignoruj filtr i skonwertuj wszystkie zadania.")
    c.add_argument("--min-rows", type=int, default=100)

    args = p.parse_args()
    if args.cmd == "download":
        cmd_download(args.root)
    elif args.cmd == "fetch":
        cmd_fetch(args.root, args.tasks, args.all_tasks, args.subjects, args.limit)
    elif args.cmd == "inspect":
        cmd_inspect(args.root, args.tasks)
    elif args.cmd == "convert":
        cmd_convert(args.root, args.out, args.tasks, args.all_tasks, args.min_rows)


if __name__ == "__main__":
    main()
