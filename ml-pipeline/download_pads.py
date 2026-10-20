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
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
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
    for base_dir in (root, *sorted(glob(os.path.join(root, "*")))):
        if not os.path.isdir(base_dir):
            continue
        movement_dir = os.path.join(base_dir, "movement")
        patients_dir = os.path.join(base_dir, "patients")
        if os.path.isdir(movement_dir) and os.path.isdir(patients_dir):
            timeseries_dir = os.path.join(movement_dir, "timeseries")
            timeseries_dir = (timeseries_dir if os.path.isdir(timeseries_dir)
                              else movement_dir)
            return {
                "base": base_dir,
                "patients": patients_dir,
                "movement": movement_dir,
                "timeseries": timeseries_dir,
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
    match = FILENAME_RE.match(os.path.basename(path))
    if not match:
        return None
    return {"subject": match.group("subject"),
            "task": match.group("task"),
            "wrist": match.group("wrist")}


def _task_matches(task: str, filters: list[str]) -> bool:
    """Czy nazwa zadania zawiera którykolwiek z filtrów (case-insensitive)?"""
    task_lower = task.lower()
    return any(task_filter.lower() in task_lower for task_filter in filters)


# ---------------------------------------------------------------------------
# Etykiety pacjentów
# ---------------------------------------------------------------------------

def _condition_to_label(condition: str) -> int:
    """
    Mapuje tekst diagnozy na klasę (detekcja PD). Heurystyka odporna na warianty
    zapisu: 0=zdrowy, 1=Parkinson, 2=inne zaburzenie ruchowe.
    """
    condition_lower = (condition or "").lower()
    if "healthy" in condition_lower or "control" in condition_lower:
        return 0
    if "parkinson" in condition_lower and "atypical" not in condition_lower:
        return 1
    return 2


def _extract_condition(flat_df: pd.DataFrame) -> str:
    """Szuka w spłaszczonych metadanych pacjenta pola z diagnozą."""
    if flat_df.empty:
        return "unknown"
    for col in flat_df.columns:
        if any(key in col.lower() for key in ("condition", "disease", "diagnos")):
            value = flat_df[col].iloc[0]
            if isinstance(value, str) and value.strip():
                return value
    return "unknown"


def _subject_from_patient_file(path: str) -> str:
    """subject_id z nazwy pliku patient_XXX.json (np. '001')."""
    match = re.search(r"(\d+)", os.path.basename(path))
    return match.group(1) if match else os.path.basename(path)


def _load_patient_labels(patients_dir: str) -> dict[str, dict]:
    """subject_id -> {condition, label} na podstawie plików patient_*.json."""
    labels_by_subject: dict[str, dict] = {}
    for path in sorted(glob(os.path.join(patients_dir, "*.json"))):
        subject_id = _subject_from_patient_file(path)
        condition = _extract_condition(flatten_json(path))
        labels_by_subject[subject_id] = {
            "condition": condition,
            "label": _condition_to_label(condition),
        }
    return labels_by_subject


# ---------------------------------------------------------------------------
# Download (opcjonalny — cały ZIP)
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Pasek postępu dla urlretrieve (ważny przy pobieraniu przez hotspot)."""
    downloaded_bytes = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded_bytes * 100.0 / total_size)
        bar = "#" * int(percent // 2.5)
        print(f"\r[download] {percent:5.1f}% "
              f"[{bar:<40}] {downloaded_bytes/1e6:7.1f} / {total_size/1e6:.1f} MB",
              end="", flush=True)
    else:
        print(f"\r[download] {downloaded_bytes/1e6:7.1f} MB", end="", flush=True)


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
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(root)
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
    subject_numbers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low_bound, high_bound = part.split("-", 1)
            subject_numbers.update(range(int(low_bound), int(high_bound) + 1))
        else:
            subject_numbers.add(int(part))
    return subject_numbers


def _fetch_manifest_paths() -> list[str]:
    """Pobiera SHA256SUMS.txt i zwraca listę ścieżek plików projektu."""
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "pads_SHA256SUMS.txt")
    print(f"[fetch] Pobieram manifest:\n  {PHYSIONET_MANIFEST}")
    urlretrieve(PHYSIONET_MANIFEST, tmp_path)
    paths: list[str] = []
    with open(tmp_path, "r", encoding="utf-8", errors="replace") as manifest_file:
        for line in manifest_file:
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
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def _download_file(rel_path: str, root: str, retries: int = 3) -> bool:
    """
    Pobiera pojedynczy plik (skip-if-exists) atomowo i z retry. Zwraca True
    przy sukcesie.

    KLUCZOWE: pobieramy do pliku tymczasowego ``.part`` i dopiero po udanym
    pobraniu robimy ``os.replace`` na docelowy ``dest``. Dzięki temu plik pod
    ``dest`` ISTNIEJE tylko wtedy, gdy jest kompletny — niepełne pobranie
    (np. „retrieval incomplete" na hotspocie) nie zostawia obciętego pliku,
    który skip-if-exists błędnie uznałby za gotowy i zepsułby ``convert``.
    """
    dest = os.path.join(root, rel_path.replace("/", os.sep))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True  # wznawialność — już pobrany (i kompletny, bo atomowy)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part_path = dest + ".part"
    url = PHYSIONET_FILES_BASE + rel_path
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            urlretrieve(url, part_path)
            os.replace(part_path, dest)  # atomowa publikacja kompletnego pliku
            return True
        except Exception as exc:  # noqa: BLE001 — raportujemy i próbujemy dalej
            last_exc = exc
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)  # sprzątamy niepełny fragment
                except OSError:
                    pass
            if attempt < retries:
                print(f"\n[fetch] Niepełne pobranie {os.path.basename(rel_path)} "
                      f"(próba {attempt}/{retries}): {exc} — ponawiam...")
    print(f"\n[fetch] BŁĄD przy {rel_path}: {last_exc}")
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
    requested_subjects = _parse_subject_spec(subjects_spec)

    manifest = _fetch_manifest_paths()

    # Sygnały timeseries pasujące do filtrów (surowe, NIE z preprocessed/).
    timeseries_paths = [path for path in manifest
                        if "timeseries/" in path and path.endswith(".txt")
                        and "preprocessed/" not in path]
    selected: list[str] = []
    for path in timeseries_paths:
        info = _parse_signal_name(path)
        if info is None:
            continue
        if not all_tasks and not _task_matches(info["task"], tasks):
            continue
        subject = _subject_of_path(path)
        if requested_subjects is not None and subject not in requested_subjects:
            continue
        selected.append(path)

    selected.sort()
    if limit is not None:
        selected = selected[:limit]

    subjects_present = sorted(
        {_subject_of_path(path) for path in selected if _subject_of_path(path)})
    subjects_present_set = set(subjects_present)
    patient_paths = [path for path in manifest
                     if "patients/" in path and path.endswith(".json")
                     and "preprocessed/" not in path
                     and _subject_of_path(path) in subjects_present_set]

    tasks_present = sorted({_parse_signal_name(path)["task"] for path in selected})
    print(f"[fetch] Filtr zadań: {'(wszystkie)' if all_tasks else tasks}")
    print(f"[fetch] Dopasowane zadania: {tasks_present}")
    print(f"[fetch] Pacjentów: {len(subjects_present)}, "
          f"sygnałów do pobrania: {len(selected)}, "
          f"plików pacjentów: {len(patient_paths)}")

    files_to_download = selected + patient_paths
    n_downloaded = 0
    for file_idx, rel_path in enumerate(files_to_download, 1):
        print(f"\r[fetch] {file_idx}/{len(files_to_download)}  "
              f"{os.path.basename(rel_path):<40}",
              end="", flush=True)
        if _download_file(rel_path, root):
            n_downloaded += 1
    print(f"\n[fetch] Pobrano {n_downloaded}/{len(files_to_download)} plików -> {root}")
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
        with open(patient_files[0], "r", encoding="utf-8") as patient_file:
            raw = json.load(patient_file)
        print(f"[inspect] Surowe klucze patient_*.json: {list(raw.keys())}")
        flat_df = flatten_json(patient_files[0])
        print(f"[inspect] Kolumny po spłaszczeniu: {list(flat_df.columns)}")
        condition = _extract_condition(flat_df)
        print(f"[inspect] Wykryta diagnoza (condition): '{condition}' "
              f"-> klasa {_condition_to_label(condition)}")

    labels_by_subject = _load_patient_labels(layout["patients"])
    if labels_by_subject:
        label_distribution: dict[int, int] = {}
        for info in labels_by_subject.values():
            label_distribution[info["label"]] = label_distribution.get(info["label"], 0) + 1
        print(f"[inspect] Rozkład etykiet pacjentów: {dict(sorted(label_distribution.items()))}")

    # --- Sygnały ---
    signals = _list_signal_files(layout["timeseries"])
    print(f"\n[inspect] Plików sygnałów (.txt): {len(signals)}")
    parsed_names = [info for info in (_parse_signal_name(sig_path) for sig_path in signals)
                    if info]
    unmatched = len(signals) - len(parsed_names)
    if unmatched:
        print(f"[inspect] UWAGA: {unmatched} plików nie pasuje do wzorca "
              f"<subject>_<task>_<wrist>.txt")

    detected_tasks = sorted({info["task"] for info in parsed_names})
    detected_wrists = sorted({info["wrist"] for info in parsed_names})
    print(f"[inspect] Wykryte nadgarstki: {detected_wrists}")
    print(f"[inspect] Wykryte zadania ({len(detected_tasks)}): {detected_tasks}")
    matching_tasks = [task for task in detected_tasks if _task_matches(task, tasks)]
    print(f"[inspect] Zadania pasujące do filtra {tasks}: {matching_tasks}")

    if signals:
        first_signal = signals[0]
        signal_array = np.loadtxt(first_signal, dtype=np.float32, delimiter=",")
        print(f"\n[inspect] Pierwszy sygnał: {os.path.basename(first_signal)}")
        print(f"[inspect]   shape={signal_array.shape} (wiersze=próbki, kolumny=kanały)")
        if signal_array.ndim == 2:
            n_columns = signal_array.shape[1]
            print(f"[inspect]   liczba kolumn: {n_columns}")
            if n_columns == len(TARGET_CHANNELS) + 1:
                print("[inspect]   -> 7 kolumn = [Time, Accel X/Y/Z, Gyro X/Y/Z] "
                      "(potwierdzone wg pads-project). Kol. 0 (czas) zostanie użyta "
                      "jako timestamp, kol. 1-6 jako kanały.")
            elif n_columns == len(TARGET_CHANNELS):
                print("[inspect]   -> 6 kolumn = [Accel X/Y/Z, Gyro X/Y/Z] "
                      "(brak kolumny czasu, timestamp syntetyczny z fs).")
            else:
                print(f"[inspect]   UWAGA: nietypowa liczba kolumn (oczekiwane 6/7).")
            print(f"[inspect]   pierwsze 2 wiersze:\n{signal_array[:2]}")


# ---------------------------------------------------------------------------
# Convert — PADS -> nasz CSV
# ---------------------------------------------------------------------------

def _load_signal(path: str) -> np.ndarray | None:
    """Wczytuje .txt (delimiter ','); zwraca macierz (próbki, kanały) lub None."""
    signal_array = np.loadtxt(path, dtype=np.float32, delimiter=",")
    if signal_array.ndim == 1:
        signal_array = signal_array[:, None]
    return signal_array


def _split_time_channels(signal_array: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Rozdziela surową macierz PADS na (timestamp, kanały).

    Autorytatywna kolejność kolumn (pads-project/run_preprocessing.py):
        [Time, Accelerometer_X/Y/Z, Gyroscope_X/Y/Z]  -> 7 kolumn.

    Zwraca:
        - 7 kolumn: kolumna 0 jako timestamp [s], kolumny 1..6 jako kanały.
        - 6 kolumn (brak czasu): timestamp syntetyczny z PADS_FS, kolumny 0..5.
        - inaczej: None (zły format).
    """
    n_channels = len(TARGET_CHANNELS)  # 6
    if signal_array.ndim != 2:
        return None
    if signal_array.shape[1] == n_channels + 1:   # [Time, accel xyz, gyro xyz]
        return signal_array[:, 0], signal_array[:, 1:1 + n_channels]
    if signal_array.shape[1] == n_channels:       # brak kolumny czasu
        return (np.arange(signal_array.shape[0]) / PADS_FS,
                signal_array[:, :n_channels])
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

    labels_by_subject = _load_patient_labels(layout["patients"])
    print(f"[convert] Wczytano etykiety dla {len(labels_by_subject)} pacjentów.")

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

        signal_array = _load_signal(path)
        time_and_channels = (_split_time_channels(signal_array)
                             if signal_array is not None else None)
        if time_and_channels is None:
            print(f"[convert] POMINIĘTO {os.path.basename(path)}: "
                  f"kształt {None if signal_array is None else signal_array.shape} "
                  f"(oczekiwane 6 lub 7 kolumn).")
            n_skipped_shape += 1
            continue
        timestamps, channel_data = time_and_channels
        if channel_data.shape[0] < min_rows:
            n_skipped_shape += 1
            continue

        # channel_data: kolumny w kolejności accel X/Y/Z, gyro X/Y/Z (TARGET_CHANNELS).
        frame = {"timestamp": timestamps}
        for channel_idx, channel in enumerate(TARGET_CHANNELS):
            frame[channel] = channel_data[:, channel_idx]
        signal_df = pd.DataFrame(frame)

        subject_id = info["subject"]
        label_info = labels_by_subject.get(
            subject_id, {"condition": "unknown", "label": -1})
        if label_info["label"] < 0:
            # Brak patient_*.json dla tego pacjenta -> nie da się trenować.
            missing_label_subjects.add(subject_id)
            n_skipped_nolabel += 1
            continue
        fname = f"{subject_id}_{info['task']}_{info['wrist']}.csv"
        signal_df.to_csv(os.path.join(out_dir, fname), index=False)
        label_rows.append({
            "filename": fname,
            "subject_id": subject_id,
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
        label_distribution: dict[int, int] = {}
        for row in label_rows:
            label_distribution[row["label"]] = label_distribution.get(row["label"], 0) + 1
        print(f"[convert] Rozkład etykiet (okna CSV): {dict(sorted(label_distribution.items()))} "
              f"(0=zdrowy, 1=PD, 2=inne, -1=brak etykiety)")
    print(f"[convert] Etykiety -> {labels_csv}")
    print(f"[convert] Dalej: python preprocess.py --input {out_dir} "
          f"--output data/processed --fs {int(PADS_FS)} --labels {labels_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptacja zbioru PADS -> nasz CSV.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    download_parser = subparsers.add_parser(
        "download", help="(Opcjonalnie) pobierz CAŁY ZIP z PhysioNet.")
    download_parser.add_argument("--root", default="data/raw")

    fetch_parser = subparsers.add_parser(
        "fetch", help="Pobierz TYLKO wybrane zadania/pacjentów z PhysioNet.")
    fetch_parser.add_argument("--root", default="data/raw/pads_dataset")
    fetch_parser.add_argument("--tasks", nargs="+", default=TREMOR_TASKS,
                              help="Filtr zadań (podłańcuchy). Domyślnie: Relaxed, StretchHold.")
    fetch_parser.add_argument("--all-tasks", action="store_true",
                              help="Pobierz wszystkie zadania (ignoruj filtr).")
    fetch_parser.add_argument("--subjects", default=None,
                              help="Pacjenci, np. '1-50', '1,2,3', '1-10,20'. Domyślnie: wszyscy.")
    fetch_parser.add_argument("--limit", type=int, default=None,
                              help="Maks. liczba plików sygnałów do pobrania (test).")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Wypisz realną strukturę / zadania / kanały.")
    inspect_parser.add_argument("--root", default="data/raw")
    inspect_parser.add_argument("--tasks", nargs="+", default=TREMOR_TASKS,
                                help="Filtr zadań (podłańcuchy). Domyślnie zadania drżenia.")

    convert_parser = subparsers.add_parser(
        "convert", help="Konwertuj wybrane zadania -> nasz CSV.")
    convert_parser.add_argument("--root", default="data/raw")
    convert_parser.add_argument("--out", default="data/raw/pads_csv")
    convert_parser.add_argument("--tasks", nargs="+", default=TREMOR_TASKS,
                                help="Filtr zadań (podłańcuchy). Domyślnie: Relaxed, StretchHold.")
    convert_parser.add_argument("--all-tasks", action="store_true",
                                help="Zignoruj filtr i skonwertuj wszystkie zadania.")
    convert_parser.add_argument("--min-rows", type=int, default=100)

    args = parser.parse_args()
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
