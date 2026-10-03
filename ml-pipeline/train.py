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
    1. Wczytanie okien (X) i etykiet (y).
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
    Wczytuje okna z pliku .npz i zwraca (X, y, subjects).

    X: (n_okien, n_probek, n_kanalow) — wprost z preprocess.py.
    y: (n_okien,) — etykiety klasy (detekcja PD: 0=zdrowy, 1=Parkinson, 2=inne).
    subjects: (n_okien,) — subject_id dla splitu per-pacjent (lub puste).

    Etykiety i subject_id pochodzą z labels.csv (download_pads.py) i są
    dołączane przez preprocess.py (flaga --labels). Jeśli ich brak, generujemy
    losowe etykiety WYŁĄCZNIE do developerskiego testu szkieletu.
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)

    if "y" in data.files:
        y = data["y"].astype(np.int64)
    else:
        print("[train] UWAGA: brak etykiet 'y' w pliku .npz — generuję losowe "
              "etykiety (placeholder developerski!). Uruchom preprocess.py z --labels.")
        rng = np.random.default_rng(42)
        y = rng.integers(low=0, high=3, size=X.shape[0]).astype(np.int64)

    subjects = (
        data["subjects"].astype(str) if "subjects" in data.files
        else np.array(["unknown"] * X.shape[0])
    )
    return X, y, subjects


def standardize(X_train: np.ndarray, X_val: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standaryzacja z-score per kanał, dopasowana wyłącznie na zbiorze treningowym
    (zapobiega wyciekowi danych). Zwraca też (mean, std) do zapisania —
    aplikacja mobilna musi stosować te same statystyki przed inferencją.
    """
    # Statystyki liczone po wymiarach (okno, czas), osobno dla każdego kanału.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    return X_train_n, X_val_n, mean.squeeze(), std.squeeze()


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


def train_model(model: keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int, batch_size: int) -> keras.callbacks.History:
    """Trenuje model z early stopping i redukcją LR."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
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
    with open(norm_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": np.asarray(mean).ravel().tolist(),
                "std": np.asarray(std).ravel().tolist(),
                "channels": [str(c) for c in channels],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[train] Statystyki normalizacji (JSON) -> {norm_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trening LSTM detekcji drżenia PD.")
    p.add_argument("--data", default="data/processed/windows.npz",
                   help="Plik .npz z preprocess.py.")
    p.add_argument("--out", default="models", help="Folder na artefakty modelu.")
    p.add_argument("--num-classes", type=int, default=3,
                   help="Liczba klas (detekcja PD: 0=zdrowy, 1=Parkinson, 2=inne).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Dane
    X, y, subjects = load_dataset(args.data)
    print(f"[train] Dane: X={X.shape}, y={y.shape}, klasy={np.unique(y)}, "
          f"pacjentów={len(set(subjects))}")

    # 2. Split train/val PER-PACJENT — okna tego samego pacjenta nie mogą
    #    trafić jednocześnie do train i val (inaczej wyciek i zawyżone wyniki).
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    if len(unique_subjects) > 1:
        rng = np.random.default_rng(args.seed)
        shuffled = rng.permutation(unique_subjects)
        n_val_subj = max(1, int(len(shuffled) * args.val_split))
        val_subjects = set(shuffled[:n_val_subj].tolist())
        val_mask = np.array([s in val_subjects for s in subjects])
        print(f"[train] Split per-pacjent: {len(val_subjects)} pacjentów do val, "
              f"{len(shuffled) - len(val_subjects)} do train.")
    else:
        # Brak informacji o pacjentach (np. tryb developerski) — split losowy.
        print("[train] UWAGA: brak subject_id — split losowy po oknach.")
        rng = np.random.default_rng(args.seed)
        val_mask = rng.random(X.shape[0]) < args.val_split

    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    X_train, X_val, mean, std = standardize(X_train, X_val)

    # 3. Model
    input_shape = (X.shape[1], X.shape[2])
    model = build_lstm_model(input_shape, args.num_classes)
    model.summary()

    # 4. Trening
    train_model(model, X_train, y_train, X_val, y_val,
                epochs=args.epochs, batch_size=args.batch_size)

    # 5. Eksport
    data = np.load(args.data, allow_pickle=True)
    channels = list(data["channels"]) if "channels" in data.files else []
    save_artifacts(model, args.out, mean, std, channels)
    export_tfjs(model, args.out)

    print("\n[train] Gotowe. Skopiuj zawartość models/tfjs_model/ "
          "do mobile-app/ (publiczny katalog assetów).")


if __name__ == "__main__":
    main()
