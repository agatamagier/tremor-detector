"""
train_freq.py
=============
Alternatywne podejście do klasyfikacji drżenia PD: zamiast podawać modelowi
surowe okno sygnału (jak robi to ``train.py`` z siecią LSTM), liczymy dla
każdego okna **cechy częstotliwościowe** (widmo mocy, PSD metodą Welcha)
i trenujemy na nich mały klasyfikator MLP.

Motywacja (dlaczego to powinno działać lepiej):
    Drżenie parkinsonowskie ma charakterystyczną częstotliwość ~4-6 Hz.
    Surowy LSTM musi sam „odkryć" tę częstotliwość z ciągu próbek góra-dół
    i mu się to nie udaje (model zapada się do jednej klasy). Jawne policzenie
    udziału energii w paśmie 4-6 Hz vs reszta pasma daje modelowi wprost tę
    informację, która różnicuje PD od zdrowych / innych zaburzeń ruchu.

Pipeline:
    1. Wczytanie okien z ``windows.npz`` (ten sam plik co dla train.py).
    2. Ekstrakcja cech PSD per kanał (6 kanałów IMU) -> wektor cech / okno.
    3. Szybki test rozdzielności: średni udział pasma 4-6 Hz PD vs nie-PD.
    4. Split per-pacjent, standaryzacja cech (z-score na train).
    5. Trening MLP z balansowaniem klas + ewaluacja per-klasa.

Użycie (CLI):
    python train_freq.py --data data/processed/windows_2class.npz \
        --out models_freq --num-classes 2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.signal import welch

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Pasma analizy [Hz].
TREMOR_BAND = (3.0, 12.0)   # całe pasmo drżenia (zgodne z filtrem w preprocess.py)
PD_BAND = (4.0, 6.0)        # charakterystyczne pasmo drżenia parkinsonowskiego
HIGH_BAND = (6.0, 12.0)     # pasmo wyższe (częstsze w drżeniu samoistnym / innym)

# Nazwy cech liczonych dla KAŻDEGO kanału (do czytelnego raportu / zapisu).
FEATURE_NAMES_PER_CHANNEL = [
    "total_power_3_12",   # łączna moc w paśmie drżenia
    "power_4_6",          # moc bezwzględna w paśmie parkinsonowskim
    "ratio_4_6",          # udział pasma 4-6 Hz w całej mocy drżenia (0-1)
    "ratio_6_12",         # udział pasma 6-12 Hz w całej mocy drżenia (0-1)
    "dominant_freq",      # częstotliwość o największej mocy w paśmie 3-12 Hz
    "spectral_centroid",  # „środek ciężkości" widma w paśmie 3-12 Hz
]


# ---------------------------------------------------------------------------
# Ekstrakcja cech częstotliwościowych
# ---------------------------------------------------------------------------

def _band_power(freqs: np.ndarray, psd: np.ndarray,
                low: float, high: float) -> float:
    """Moc (całka PSD) w paśmie [low, high] Hz metodą trapezów."""
    band_mask = (freqs >= low) & (freqs <= high)
    if not band_mask.any():
        return 0.0
    return float(np.trapz(psd[band_mask], freqs[band_mask]))


def extract_window_features(window: np.ndarray, fs: float) -> np.ndarray:
    """
    Liczy wektor cech częstotliwościowych dla jednego okna sygnału.

    window: (n_probek, n_kanalow) — okno wprost z preprocess.py.
    Zwraca: wektor długości n_kanalow * len(FEATURE_NAMES_PER_CHANNEL).
    """
    n_channels = window.shape[1]
    nperseg = min(window.shape[0], 256)
    features: list[float] = []

    for channel_idx in range(n_channels):
        freqs, psd = welch(window[:, channel_idx], fs=fs, nperseg=nperseg)

        band_mask = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
        band_freqs = freqs[band_mask]
        band_psd = psd[band_mask]
        total_power = _band_power(freqs, psd, *TREMOR_BAND) + 1e-12

        power_pd = _band_power(freqs, psd, *PD_BAND)
        power_high = _band_power(freqs, psd, *HIGH_BAND)

        if band_psd.size:
            dominant_freq = float(band_freqs[np.argmax(band_psd)])
            spectral_centroid = float(np.sum(band_freqs * band_psd) / total_power)
        else:
            dominant_freq = 0.0
            spectral_centroid = 0.0

        features.extend([
            total_power,
            power_pd,
            power_pd / total_power,
            power_high / total_power,
            dominant_freq,
            spectral_centroid,
        ])

    return np.asarray(features, dtype=np.float32)


def build_feature_matrix(windows: np.ndarray, fs: float) -> np.ndarray:
    """Liczy macierz cech (n_okien, n_cech) dla wszystkich okien."""
    return np.stack([extract_window_features(window, fs) for window in windows])


# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------

def load_dataset(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Wczytuje okna, etykiety, subject_id i fs z pliku .npz (z preprocess.py)."""
    data = np.load(npz_path, allow_pickle=True)
    windows = data["X"].astype(np.float32)
    labels = data["y"].astype(np.int64)
    subjects = (data["subjects"].astype(str) if "subjects" in data.files
                else np.array(["unknown"] * windows.shape[0]))
    fs = float(data["fs"]) if "fs" in data.files else 100.0
    return windows, labels, subjects, fs


def sanity_check_pd_band(features: np.ndarray, labels: np.ndarray,
                         n_channels: int) -> None:
    """
    Szybki test rozdzielności BEZ modelu: czy udział pasma 4-6 Hz (cecha
    'ratio_4_6', uśredniona po kanałach) faktycznie różni się między PD a resztą?
    Jeśli nie — żadna liczba klas ani długość okna tego nie uratuje.
    """
    n_feats_per_ch = len(FEATURE_NAMES_PER_CHANNEL)
    ratio_idx = FEATURE_NAMES_PER_CHANNEL.index("ratio_4_6")
    # Kolumny 'ratio_4_6' dla każdego kanału -> średnia po kanałach / okno.
    cols = [channel_idx * n_feats_per_ch + ratio_idx for channel_idx in range(n_channels)]
    ratio_46_mean = features[:, cols].mean(axis=1)

    print("\n[sanity] Udział energii w paśmie 4-6 Hz (parkinsonowskim):")
    for class_value in sorted(set(labels.tolist())):
        values = ratio_46_mean[labels == class_value]
        print(f"  klasa {class_value}: średnia={values.mean():.3f}  "
              f"mediana={np.median(values):.3f}  (n={len(values)})")
    print("  (jeśli PD ma wyraźnie wyższy udział 4-6 Hz niż reszta -> cecha ma sens)")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_mlp(n_features: int, num_classes: int) -> keras.Model:
    """Mały MLP na cechach tabelarycznych (szybki, odporny przy ~70 pacjentach)."""
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,), name="freq_features"),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax", name="diagnosis"),
        ],
        name="tremor_freq_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(train_labels: np.ndarray, num_classes: int) -> dict[int, float]:
    """Wagi klas odwrotnie proporcjonalne do liczności (balanced), jak sklearn."""
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    n_total = counts.sum()
    weights: dict[int, float] = {}
    for class_idx in range(num_classes):
        weights[class_idx] = (n_total / (num_classes * counts[class_idx])
                              if counts[class_idx] > 0 else 1.0)
    return weights


def evaluate_per_class(model: keras.Model, val_features: np.ndarray,
                       val_labels: np.ndarray, num_classes: int) -> float:
    """Macierz pomyłek + precision/recall/F1 per klasa + balanced accuracy."""
    if len(val_labels) == 0:
        print("[eval] Pusty zbiór walidacyjny — pomijam.")
        return 0.0

    predicted_labels = model.predict(val_features, verbose=0).argmax(axis=1)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(val_labels, predicted_labels):
        confusion_matrix[int(true_label), int(predicted_label)] += 1

    print("\n[eval] Macierz pomyłek (wiersze = prawda, kolumny = predykcja):")
    print("        " + "".join(f"pred{class_idx:<6}" for class_idx in range(num_classes)))
    for class_idx in range(num_classes):
        row = "".join(f"{confusion_matrix[class_idx, pred_idx]:<10}"
                      for pred_idx in range(num_classes))
        print(f"  true{class_idx}  {row}")

    print("\n[eval] Metryki per-klasa:")
    print("  klasa   precision  recall    f1       support")
    for class_idx in range(num_classes):
        true_pos = confusion_matrix[class_idx, class_idx]
        false_pos = confusion_matrix[:, class_idx].sum() - true_pos
        false_neg = confusion_matrix[class_idx, :].sum() - true_pos
        support = confusion_matrix[class_idx, :].sum()
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        print(f"  {class_idx:<7} {precision:<10.3f} {recall:<9.3f} {f1:<8.3f} {support}")

    per_class_recall = [
        confusion_matrix[class_idx, class_idx] / confusion_matrix[class_idx, :].sum()
        if confusion_matrix[class_idx, :].sum() > 0 else 0.0
        for class_idx in range(num_classes)
    ]
    balanced_accuracy = float(np.mean(per_class_recall))
    print(f"\n[eval] Balanced accuracy (średni recall): {balanced_accuracy:.3f}")
    return balanced_accuracy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening klasyfikatora drżenia PD na cechach częstotliwościowych.")
    parser.add_argument("--data", default="data/processed/windows.npz")
    parser.add_argument("--out", default="models_freq")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Dane + ekstrakcja cech
    windows, labels, subjects, fs = load_dataset(args.data)
    valid_mask = labels >= 0
    if not valid_mask.all():
        windows, labels, subjects = (windows[valid_mask], labels[valid_mask],
                                     subjects[valid_mask])
    n_channels = windows.shape[2]
    print(f"[freq] Ekstrakcja cech PSD: {windows.shape[0]} okien x {n_channels} kanałów "
          f"(fs={fs} Hz)...")
    features = build_feature_matrix(windows, fs)
    print(f"[freq] Macierz cech: {features.shape} "
          f"({len(FEATURE_NAMES_PER_CHANNEL)} cech/kanał x {n_channels} kanałów)")
    print(f"[freq] Klasy={np.unique(labels)}, pacjentów={len(set(subjects))}")

    # 2. Test rozdzielności pasma 4-6 Hz (bez modelu)
    sanity_check_pd_band(features, labels, n_channels)

    # 3. Split per-pacjent (bez wycieku okien jednego pacjenta do train i val)
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique_subjects)
    n_val = max(1, int(len(shuffled) * args.val_split))
    val_subjects = set(shuffled[:n_val].tolist())
    val_mask = np.array([subject in val_subjects for subject in subjects])
    print(f"\n[freq] Split per-pacjent: {len(val_subjects)} val / "
          f"{len(shuffled) - len(val_subjects)} train.")

    train_features, train_labels = features[~val_mask], labels[~val_mask]
    val_features, val_labels = features[val_mask], labels[val_mask]

    # 4. Standaryzacja cech (z-score liczony tylko na train)
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-8
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std

    # 5. Model + trening
    model = build_mlp(features.shape[1], args.num_classes)
    model.summary()

    if args.no_class_weights:
        class_weight = None
    else:
        class_weight = compute_class_weights(train_labels, args.num_classes)
        print(f"[freq] Wagi klas (balanced): "
              f"{ {c: round(w, 3) for c, w in class_weight.items()} }")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                      patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=5, min_lr=1e-5),
    ]
    model.fit(train_features, train_labels,
              validation_data=(val_features, val_labels),
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=callbacks, class_weight=class_weight, verbose=2)

    # 6. Ewaluacja
    evaluate_per_class(model, val_features, val_labels, args.num_classes)

    # 7. Zapis artefaktów
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "tremor_freq_mlp.keras"))
    np.savez(os.path.join(args.out, "feature_norm.npz"),
             mean=mean.squeeze(), std=std.squeeze())
    with open(os.path.join(args.out, "feature_spec.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_names_per_channel": FEATURE_NAMES_PER_CHANNEL,
            "n_channels": n_channels,
            "fs": fs,
            "bands": {"tremor": TREMOR_BAND, "pd": PD_BAND, "high": HIGH_BAND},
            "mean": mean.squeeze().tolist(),
            "std": std.squeeze().tolist(),
        }, f, ensure_ascii=False, indent=2)
    print(f"[freq] Zapisano model + specyfikację cech -> {args.out}")


if __name__ == "__main__":
    main()
