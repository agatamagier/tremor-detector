"""
train.py
========
Szkielet treningu modelu LSTM do klasyfikacji nasilenia drżenia
parkinsonowskiego na podstawie okien sygnału IMU wygenerowanych przez
``preprocess.py`` (data/processed/windows.npz).

Model przewiduje poziom nasilenia drżenia w skali porządkowej, np.:
    0 = brak / minimalne, 1 = łagodne, 2 = umiarkowane, 3 = silne
(mapowanie można dostosować do etykiet MDS-UPDRS / konkretnej bazy).

Pipeline:
    1. Wczytanie okien (windows) i etykiet (labels).
    2. Standaryzacja per-kanał (z-score) + split train/val.
    3. Budowa i trening sieci LSTM.
    4. Eksport:
       - natywny format Keras (.keras) do dalszych iteracji,
       - TensorFlow.js (model.json + *.bin) dla aplikacji mobilnej.

UWAGA: przypisanie etykiet (``load_dataset``) jest celowo zostawione jako
punkt do uzupełnienia — zależy od schematu metadanych mPower/PADS.

Użycie (CLI):
    python train.py --data data/processed/windows.npz \
        --out models --epochs 50 --num-classes 4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# TensorFlow / Keras importujemy wewnątrz funkcji tam, gdzie to możliwe,
# aby samo wczytanie modułu (np. do testów preprocessingu) nie wymagało TF.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------

def load_dataset(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wczytuje okna z pliku .npz i zwraca (windows, labels, subjects).

    windows:  (n_okien, n_probek, n_kanalow) — wprost z preprocess.py.
    labels:   (n_okien,) — etykiety klasy (detekcja PD: 0=zdrowy, 1=Parkinson, 2=inne).
    subjects: (n_okien,) — subject_id dla splitu per-pacjent (lub puste).

    Etykiety i subject_id pochodzą z labels.csv (download_pads.py) i są
    dołączane przez preprocess.py (flaga --labels). Jeśli ich brak, generujemy
    losowe etykiety WYŁĄCZNIE do developerskiego testu szkieletu.
    """
    data = np.load(npz_path, allow_pickle=True)
    windows = data["X"].astype(np.float32)

    if "y" in data.files:
        labels = data["y"].astype(np.int64)
    else:
        print("[train] UWAGA: brak etykiet 'y' w pliku .npz — generuję losowe "
              "etykiety (placeholder developerski!). Uruchom preprocess.py z --labels.")
        rng = np.random.default_rng(42)
        labels = rng.integers(low=0, high=3, size=windows.shape[0]).astype(np.int64)

    subjects = (
        data["subjects"].astype(str) if "subjects" in data.files
        else np.array(["unknown"] * windows.shape[0])
    )
    return windows, labels, subjects


def standardize(train_windows: np.ndarray, val_windows: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standaryzacja z-score per kanał, dopasowana wyłącznie na zbiorze treningowym
    (zapobiega wyciekowi danych). Zwraca też (mean, std) do zapisania —
    aplikacja mobilna musi stosować te same statystyki przed inferencją.
    """
    # Statystyki liczone po wymiarach (okno, czas), osobno dla każdego kanału.
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True) + 1e-8

    train_windows_norm = (train_windows - mean) / std
    val_windows_norm = (val_windows - mean) / std
    return train_windows_norm, val_windows_norm, mean.squeeze(), std.squeeze()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_lstm_model(input_shape: tuple[int, int],
                     num_classes: int) -> keras.Model:
    """
    Buduje sieć LSTM dla sekwencji czasowych IMU.

    input_shape: (n_probek_w_oknie, n_kanalow), np. (150, 6).

    Architektura dobrana pod Edge AI (mały, szybki model do konwersji na
    TensorFlow.js i inferencji na telefonie):
        - Maskowanie/normalizacja wejścia odbywa się poza modelem (standardize).
        - Dwuwarstwowy stos LSTM (64 -> 32) wychwytuje dynamikę drżenia.
        - Dropout dla regularyzacji przy ograniczonych danych klinicznych.
        - Softmax dla klasyfikacji wieloklasowej nasilenia.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="imu_window"),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax", name="severity"),
        ],
        name="tremor_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(train_labels: np.ndarray, num_classes: int
                          ) -> dict[int, float]:
    """
    Wagi klas odwrotnie proporcjonalne do liczności (balanced), tak jak
    sklearn: w_c = n_total / (n_klas * n_c). PADS jest niezbalansowany
    (dużo „Parkinson's", mało „inne"), więc bez tego model zawyża klasę
    większościową. Klasy nieobecne w train dostają wagę 1.0 (neutralnie).
    """
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    n_total = counts.sum()
    weights: dict[int, float] = {}
    for class_idx in range(num_classes):
        if counts[class_idx] > 0:
            weights[class_idx] = n_total / (num_classes * counts[class_idx])
        else:
            weights[class_idx] = 1.0
    return weights


def train_model(model: keras.Model,
                train_windows: np.ndarray, train_labels: np.ndarray,
                val_windows: np.ndarray, val_labels: np.ndarray,
                epochs: int, batch_size: int,
                class_weight: dict[int, float] | None = None
                ) -> keras.callbacks.History:
    """Trenuje model z early stopping, redukcją LR i (opcjonalnie) wagami klas."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]
    return model.fit(
        train_windows, train_labels,
        validation_data=(val_windows, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )


# ---------------------------------------------------------------------------
# Eksport
# ---------------------------------------------------------------------------

def export_tfjs(model: keras.Model, out_dir: str) -> None:
    """
    Eksportuje model do formatu TensorFlow.js (model.json + wagi *.bin),
    który aplikacja Vue/Capacitor ładuje z lokalnego pliku.

    Wymaga pakietu ``tensorflowjs`` (patrz requirements.txt).
    """
    try:
        import tensorflowjs as tfjs
    except ImportError as exc:
        raise ImportError(
            "Brak pakietu 'tensorflowjs'. Zainstaluj: pip install tensorflowjs"
        ) from exc

    tfjs_dir = os.path.join(out_dir, "tfjs_model")
    os.makedirs(tfjs_dir, exist_ok=True)
    tfjs.converters.save_keras_model(model, tfjs_dir)
    print(f"[train] Model TF.js zapisany -> {tfjs_dir}/model.json")


def evaluate_per_class(model: keras.Model,
                       val_windows: np.ndarray, val_labels: np.ndarray,
                       num_classes: int) -> None:
    """
    Macierz pomyłek + precision/recall/F1 per klasa na zbiorze walidacyjnym.
    Przy niezbalansowanych danych to ważniejsze niż sama accuracy — pokazuje,
    czy model faktycznie wykrywa klasy mniejszościowe. Tylko numpy (bez sklearn).
    """
    if len(val_labels) == 0:
        print("[eval] Pusty zbiór walidacyjny — pomijam ewaluację per-klasa.")
        return

    predicted_labels = model.predict(val_windows, verbose=0).argmax(axis=1)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(val_labels, predicted_labels):
        confusion_matrix[int(true_label), int(predicted_label)] += 1

    print("\n[eval] Macierz pomyłek (wiersze = prawda, kolumny = predykcja):")
    header = "        " + "".join(f"pred{class_idx:<6}" for class_idx in range(num_classes))
    print(header)
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

    # Balanced accuracy = średni recall po klasach (odporne na niezbalansowanie).
    per_class_recall = [
        confusion_matrix[class_idx, class_idx] / confusion_matrix[class_idx, :].sum()
        if confusion_matrix[class_idx, :].sum() > 0 else 0.0
        for class_idx in range(num_classes)
    ]
    print(f"\n[eval] Balanced accuracy (średni recall): {np.mean(per_class_recall):.3f}")


def save_artifacts(model: keras.Model, out_dir: str,
                   mean: np.ndarray, std: np.ndarray,
                   channels: list[str]) -> None:
    """Zapisuje model Keras, statystyki normalizacji i metadane inferencji."""
    os.makedirs(out_dir, exist_ok=True)

    keras_path = os.path.join(out_dir, "tremor_lstm.keras")
    model.save(keras_path)
    print(f"[train] Model Keras zapisany -> {keras_path}")

    # Statystyki normalizacji + metadane — aplikacja mobilna MUSI stosować
    # dokładnie te same wartości mean/std przed inferencją.
    norm_path = os.path.join(out_dir, "normalization.npz")
    np.savez(norm_path, mean=mean, std=std, channels=np.array(channels))
    print(f"[train] Statystyki normalizacji -> {norm_path}")

    # Wersja JSON wczytywana wprost przez aplikację mobilną (TF.js).
    norm_json = os.path.join(out_dir, "normalization.json")
    with open(norm_json, "w", encoding="utf-8") as norm_file:
        json.dump(
            {
                "mean": np.asarray(mean).ravel().tolist(),
                "std": np.asarray(std).ravel().tolist(),
                "channels": [str(channel) for channel in channels],
            },
            norm_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[train] Statystyki normalizacji (JSON) -> {norm_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening LSTM detekcji drżenia PD.")
    parser.add_argument("--data", default="data/processed/windows.npz",
                        help="Plik .npz z preprocess.py.")
    parser.add_argument("--out", default="models", help="Folder na artefakty modelu.")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Liczba klas (detekcja PD: 0=zdrowy, 1=Parkinson, 2=inne).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Wyłącz balansowanie klas (domyślnie włączone).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Dane
    windows, labels, subjects = load_dataset(args.data)

    # Odfiltruj okna bez etykiety (label < 0, np. brak patient_*.json w PADS).
    valid_mask = labels >= 0
    if not valid_mask.all():
        print(f"[train] Pomijam {int((~valid_mask).sum())} okien bez etykiety (label<0).")
        windows, labels, subjects = (
            windows[valid_mask], labels[valid_mask], subjects[valid_mask])

    print(f"[train] Dane: X={windows.shape}, y={labels.shape}, "
          f"klasy={np.unique(labels)}, pacjentów={len(set(subjects))}")

    # 2. Split train/val PER-PACJENT — okna tego samego pacjenta nie mogą
    #    trafić jednocześnie do train i val (inaczej wyciek i zawyżone wyniki).
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    if len(unique_subjects) > 1:
        rng = np.random.default_rng(args.seed)
        shuffled_subjects = rng.permutation(unique_subjects)
        n_val_subjects = max(1, int(len(shuffled_subjects) * args.val_split))
        val_subjects = set(shuffled_subjects[:n_val_subjects].tolist())
        val_mask = np.array([subject in val_subjects for subject in subjects])
        print(f"[train] Split per-pacjent: {len(val_subjects)} pacjentów do val, "
              f"{len(shuffled_subjects) - len(val_subjects)} do train.")
    else:
        # Brak informacji o pacjentach (np. tryb developerski) — split losowy.
        print("[train] UWAGA: brak subject_id — split losowy po oknach.")
        rng = np.random.default_rng(args.seed)
        val_mask = rng.random(windows.shape[0]) < args.val_split

    train_windows, train_labels = windows[~val_mask], labels[~val_mask]
    val_windows, val_labels = windows[val_mask], labels[val_mask]

    train_windows, val_windows, mean, std = standardize(train_windows, val_windows)

    # 3. Model
    input_shape = (windows.shape[1], windows.shape[2])
    model = build_lstm_model(input_shape, args.num_classes)
    model.summary()

    # 4. Trening (z balansowaniem klas, chyba że wyłączone)
    if args.no_class_weights:
        class_weight = None
        print("[train] Balansowanie klas: WYŁĄCZONE.")
    else:
        class_weight = compute_class_weights(train_labels, args.num_classes)
        print(f"[train] Wagi klas (balanced): "
              f"{ {class_idx: round(weight, 3) for class_idx, weight in class_weight.items()} }")
    train_model(model, train_windows, train_labels, val_windows, val_labels,
                epochs=args.epochs, batch_size=args.batch_size,
                class_weight=class_weight)

    # 5. Ewaluacja per-klasa (ważne przy niezbalansowanych danych)
    evaluate_per_class(model, val_windows, val_labels, args.num_classes)

    # 6. Eksport
    data = np.load(args.data, allow_pickle=True)
    channels = list(data["channels"]) if "channels" in data.files else []
    save_artifacts(model, args.out, mean, std, channels)
    try:
        export_tfjs(model, args.out)
    except ImportError as exc:
        print(f"\n[train] Pominięto eksport TF.js: {exc}")
        print("[train] Model Keras zapisany. Eksport zrób w osobnym venv "
              "(requirements-tfjs.txt):\n"
              "  tensorflowjs_converter --input_format=keras "
              f"{os.path.join(args.out, 'tremor_lstm.keras')} "
              f"{os.path.join(args.out, 'tfjs_model')}")
    else:
        print("\n[train] Gotowe. Skopiuj zawartość models/tfjs_model/ "
              "do mobile-app/ (publiczny katalog assetów).")


if __name__ == "__main__":
    main()
