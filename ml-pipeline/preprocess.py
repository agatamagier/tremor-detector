"""
preprocess.py
=============
Przetwarzanie wstępne sygnałów inercyjnych (IMU) do detekcji drżenia
parkinsonowskiego.

Pipeline:
    1. Wczytanie surowych nagrań CSV (mPower / PADS) z kolumnami:
       timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    2. Resampling do stałej częstotliwości próbkowania (jednolita siatka czasu).
    3. Filtr pasmowo-przepustowy Butterwortha 3-12 Hz (zero-phase, filtfilt),
       który usuwa składową grawitacyjną / dryf (< 3 Hz) oraz szum
       wysokoczęstotliwościowy (> 12 Hz), pozostawiając pasmo drżenia PD.
    4. Okienkowanie sygnału metodą sliding window (okna 3 s, 50% overlap).

Wynik: tensor okien o kształcie (n_okien, n_probek_w_oknie, n_kanalow)
zapisywany do ml-pipeline/data/processed/ jako .npz, gotowy do train.py.

Użycie (CLI):
    python preprocess.py --input data/raw --output data/processed \
        --fs 50 --lowcut 3 --highcut 12 --window-sec 3 --overlap 0.5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

#: Kanały sygnału, których używamy jako cechy wejściowe modelu.
SENSOR_CHANNELS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
]

#: Kolumna z czasem.
TIME_COLUMN = "timestamp"


@dataclass
class PreprocessConfig:
    """Parametry przetwarzania sygnału."""

    fs: float = 50.0            # docelowa częstotliwość próbkowania [Hz]
    lowcut: float = 3.0         # dolna granica pasma drżenia [Hz]
    highcut: float = 12.0       # górna granica pasma drżenia [Hz]
    order: int = 4              # rząd filtra Butterwortha
    window_sec: float = 3.0     # długość okna [s]
    overlap: float = 0.5        # nakładanie okien (0.5 = 50%)
    channels: list = field(default_factory=lambda: list(SENSOR_CHANNELS))

    @property
    def window_samples(self) -> int:
        """Długość okna w próbkach."""
        return int(round(self.window_sec * self.fs))

    @property
    def step_samples(self) -> int:
        """Krok przesunięcia okna w próbkach (z uwzględnieniem overlap)."""
        return max(1, int(round(self.window_samples * (1.0 - self.overlap))))


# ---------------------------------------------------------------------------
# Wczytywanie i czyszczenie
# ---------------------------------------------------------------------------

def load_csv(path: str, channels: list[str] | None = None) -> pd.DataFrame:
    """
    Wczytuje pojedynczy plik CSV z nagraniem IMU.

    Oczekiwane kolumny: timestamp + kanały czujników.
    Brakujące próbki są interpolowane liniowo (krótkie przerwy w strumieniu).
    """
    channels = channels or SENSOR_CHANNELS
    signal_df = pd.read_csv(path)

    missing_columns = [col for col in [TIME_COLUMN, *channels]
                       if col not in signal_df.columns]
    if missing_columns:
        raise ValueError(
            f"Plik {path} nie zawiera wymaganych kolumn: {missing_columns}. "
            f"Dostępne kolumny: {list(signal_df.columns)}"
        )

    signal_df = signal_df[[TIME_COLUMN, *channels]].copy()
    # Interpolacja drobnych braków, a następnie usunięcie ewentualnych
    # pozostałości na brzegach.
    signal_df[channels] = signal_df[channels].interpolate(
        method="linear", limit_direction="both")
    signal_df = signal_df.dropna().reset_index(drop=True)
    return signal_df


def resample_uniform(df: pd.DataFrame, fs: float,
                     channels: list[str] | None = None) -> np.ndarray:
    """
    Przepróbkowuje sygnał na jednolitą siatkę czasu o częstotliwości ``fs``.

    Urządzenia mobilne rzadko dostarczają idealnie równych odstępów między
    próbkami, dlatego interpolujemy sygnał na regularną oś czasu — jest to
    warunek konieczny poprawnego działania filtra cyfrowego.

    Zwraca macierz (n_probek, n_kanalow).
    """
    channels = channels or SENSOR_CHANNELS
    time_s = df[TIME_COLUMN].to_numpy(dtype=float)

    # Normalizacja czasu: jeśli timestamp jest w milisekundach, przelicz na s.
    time_s = time_s - time_s[0]
    if time_s[-1] > 1e4:  # heurystyka: wartości rzędu >10000 to prawdopodobnie ms
        time_s = time_s / 1000.0

    duration_s = time_s[-1]
    n_samples = int(np.floor(duration_s * fs)) + 1
    time_uniform = np.arange(n_samples) / fs

    resampled = np.empty((n_samples, len(channels)), dtype=np.float64)
    for channel_idx, channel in enumerate(channels):
        resampled[:, channel_idx] = np.interp(
            time_uniform, time_s, df[channel].to_numpy(dtype=float))
    return resampled


# ---------------------------------------------------------------------------
# Filtr Butterwortha (bandpass 3-12 Hz)
# ---------------------------------------------------------------------------

def design_bandpass(lowcut: float, highcut: float, fs: float,
                    order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Projektuje współczynniki (b, a) filtra pasmowo-przepustowego Butterwortha.

    ``b_coeffs`` to licznik, ``a_coeffs`` to mianownik transmitancji filtra.

    Pasmo 3-12 Hz odpowiada typowemu zakresowi drżenia w chorobie Parkinsona
    (drżenie spoczynkowe ~4-6 Hz, drżenie posturalne/kinetyczne do ~12 Hz).
    """
    nyquist = 0.5 * fs
    low_norm = lowcut / nyquist
    high_norm = highcut / nyquist
    if not (0 < low_norm < high_norm < 1):
        raise ValueError(
            f"Nieprawidłowe pasmo dla fs={fs} Hz: low={lowcut}, high={highcut}. "
            f"Wymagane 0 < lowcut < highcut < Nyquist ({nyquist} Hz)."
        )
    b_coeffs, a_coeffs = butter(order, [low_norm, high_norm], btype="band")
    return b_coeffs, a_coeffs


def apply_bandpass(signal: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """
    Stosuje zero-phase bandpass (filtfilt) na każdym kanale osobno.

    ``filtfilt`` filtruje sygnał dwukierunkowo, dzięki czemu nie wprowadza
    przesunięcia fazowego — istotne przy analizie kształtu drżenia.

    ``signal``: (n_probek, n_kanalow) -> zwraca macierz tego samego kształtu.
    """
    b_coeffs, a_coeffs = design_bandpass(
        config.lowcut, config.highcut, config.fs, config.order)

    # filtfilt wymaga, by długość sygnału była większa niż ~3 * max(len(a), len(b)).
    min_length = 3 * max(len(a_coeffs), len(b_coeffs))
    if signal.shape[0] <= min_length:
        raise ValueError(
            f"Sygnał zbyt krótki ({signal.shape[0]} próbek) dla filtra "
            f"rzędu {config.order}. Wymagane > {min_length} próbek."
        )

    filtered = np.empty_like(signal)
    for channel_idx in range(signal.shape[1]):
        filtered[:, channel_idx] = filtfilt(
            b_coeffs, a_coeffs, signal[:, channel_idx])
    return filtered


# ---------------------------------------------------------------------------
# Okienkowanie (sliding window)
# ---------------------------------------------------------------------------

def sliding_windows(signal: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """
    Dzieli sygnał na nakładające się okna.

    ``signal``: (n_probek, n_kanalow)
    Zwraca tensor (n_okien, window_samples, n_kanalow).

    Dla okna 3 s przy fs=50 Hz i 50% overlap: okno=150 próbek, krok=75 próbek.
    """
    window_len = config.window_samples
    step_len = config.step_samples
    n_samples_total = signal.shape[0]

    if n_samples_total < window_len:
        return np.empty((0, window_len, signal.shape[1]), dtype=signal.dtype)

    window_starts = range(0, n_samples_total - window_len + 1, step_len)
    windows = np.stack(
        [signal[start_idx:start_idx + window_len] for start_idx in window_starts],
        axis=0,
    )
    return windows


# ---------------------------------------------------------------------------
# Orkiestracja: plik -> okna
# ---------------------------------------------------------------------------

def process_file(path: str, config: PreprocessConfig) -> np.ndarray:
    """Pełny pipeline dla jednego pliku CSV: load -> resample -> filter -> window."""
    signal_df = load_csv(path, config.channels)
    uniform_signal = resample_uniform(signal_df, config.fs, config.channels)
    filtered_signal = apply_bandpass(uniform_signal, config)
    windows = sliding_windows(filtered_signal, config)
    return windows


def load_labels(labels_path: str) -> tuple[dict[str, int], dict[str, str]]:
    """
    Wczytuje mapowania z pliku labels.csv (produkowanego przez download_pads.py):
        filename -> label   oraz   filename -> subject_id.

    Dzięki subject_id train.py może zrobić split per-pacjent (bez wycieku okien
    tego samego pacjenta między train/val).
    """
    labels_df = pd.read_csv(labels_path)
    if "filename" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError(
            f"{labels_path} musi mieć kolumny 'filename' i 'label'. "
            f"Znaleziono: {list(labels_df.columns)}"
        )
    label_map = dict(zip(labels_df["filename"].astype(str),
                         labels_df["label"].astype(int)))
    subject_column = "subject_id" if "subject_id" in labels_df.columns else None
    subject_map = (
        dict(zip(labels_df["filename"].astype(str),
                 labels_df[subject_column].astype(str)))
        if subject_column else {}
    )
    return label_map, subject_map


def process_directory(input_dir: str, output_dir: str,
                      config: PreprocessConfig,
                      labels_path: str | None = None) -> None:
    """
    Przetwarza wszystkie pliki CSV z ``input_dir`` i zapisuje połączony
    zbiór okien do ``output_dir/windows.npz``.

    Jeśli podano ``labels_path`` (labels.csv), do każdego okna dołączana jest
    etykieta ``y`` oraz ``subject`` na podstawie nazwy pliku źródłowego —
    gotowe pod trening nadzorowany (detekcja PD) ze splitem per-pacjent.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(
        fname for fname in os.listdir(input_dir) if fname.lower().endswith(".csv")
    )
    if not csv_files:
        print(f"[preprocess] Brak plików CSV w {input_dir}. Nic do zrobienia.")
        return

    label_map: dict[str, int] = {}
    subject_map: dict[str, str] = {}
    if labels_path:
        label_map, subject_map = load_labels(labels_path)
        print(f"[preprocess] Wczytano etykiety dla {len(label_map)} plików.")

    all_windows: list[np.ndarray] = []
    source_index: list[str] = []   # ślad pochodzenia każdego okna
    label_index: list[int] = []    # etykieta każdego okna (jeśli dostępna)
    subject_index: list[str] = []  # subject_id każdego okna (jeśli dostępny)

    for fname in csv_files:
        if labels_path and fname not in label_map:
            # Plik bez etykiety w trybie nadzorowanym — pomiń (np. labels.csv sam).
            continue
        file_path = os.path.join(input_dir, fname)
        try:
            windows = process_file(file_path, config)
        except ValueError as exc:
            print(f"[preprocess] POMINIĘTO {fname}: {exc}")
            continue

        if windows.shape[0] == 0:
            print(f"[preprocess] {fname}: za mało danych na choćby jedno okno.")
            continue

        n_windows = windows.shape[0]
        all_windows.append(windows)
        source_index.extend([fname] * n_windows)
        if labels_path:
            label_index.extend([label_map[fname]] * n_windows)
            subject_index.extend([subject_map.get(fname, "unknown")] * n_windows)
        print(f"[preprocess] {fname}: {n_windows} okien "
              f"({windows.shape[1]} próbek x {windows.shape[2]} kanałów)")

    if not all_windows:
        print("[preprocess] Nie wygenerowano żadnych okien.")
        return

    windows_array = np.concatenate(all_windows, axis=0).astype(np.float32)
    arrays = dict(
        X=windows_array,
        sources=np.array(source_index),
        channels=np.array(config.channels),
        fs=config.fs,
        window_sec=config.window_sec,
        overlap=config.overlap,
    )
    if labels_path:
        arrays["y"] = np.array(label_index, dtype=np.int64)
        arrays["subjects"] = np.array(subject_index)

    out_path = os.path.join(output_dir, "windows.npz")
    np.savez_compressed(out_path, **arrays)
    print(f"\n[preprocess] Zapisano {windows_array.shape[0]} okien -> {out_path}")
    print(f"[preprocess] Kształt tensora X: {windows_array.shape}")
    if labels_path:
        classes, counts = np.unique(arrays["y"], return_counts=True)
        print(f"[preprocess] Rozkład klas: {dict(zip(classes.tolist(), counts.tolist()))}")
        print(f"[preprocess] Liczba pacjentów: {len(set(subject_index))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing IMU dla detekcji drżenia PD.")
    parser.add_argument("--input", default="data/raw", help="Folder z surowymi CSV.")
    parser.add_argument("--output", default="data/processed", help="Folder wyjściowy .npz.")
    parser.add_argument("--fs", type=float, default=50.0, help="Docelowa fs [Hz].")
    parser.add_argument("--lowcut", type=float, default=3.0, help="Dolna granica pasma [Hz].")
    parser.add_argument("--highcut", type=float, default=12.0, help="Górna granica pasma [Hz].")
    parser.add_argument("--order", type=int, default=4, help="Rząd filtra Butterwortha.")
    parser.add_argument("--window-sec", type=float, default=3.0, help="Długość okna [s].")
    parser.add_argument("--overlap", type=float, default=0.5, help="Nakładanie okien (0-1).")
    parser.add_argument("--labels", default=None,
                        help="Opcjonalny labels.csv (filename,label[,subject_id]) do treningu nadzorowanego.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessConfig(
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.order,
        window_sec=args.window_sec,
        overlap=args.overlap,
    )
    print(f"[preprocess] Konfiguracja: {config}")
    print(f"[preprocess] Okno = {config.window_samples} próbek, "
          f"krok = {config.step_samples} próbek")
    process_directory(args.input, args.output, config, labels_path=args.labels)


if __name__ == "__main__":
    main()
