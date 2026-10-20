"""
kfold_freq.py
=============
Walidacja **k-krotna grupowa (po pacjencie)** dla klasyfikatora cech
częstotliwościowych (train_freq.py). Zastępuje pojedynczy split per-pacjent,
który przy ~5-17 pacjentach walidacyjnych jest „loterią" (patrz DZIENNIK,
ograniczenie smoke-testu eksperymentu 7).

Zasada:
    - podział na foldy po PACJENTACH (żaden pacjent nie jest i w train, i w val);
    - w każdym foldzie: standaryzacja cech na train, świeży MLP, ewaluacja
      okno-po-oknie ORAZ z agregacją per-pacjent (uśrednione prawdopodobieństwo);
    - raport: balanced accuracy i recall PD per fold + średnia ± odch. std,
      zbiorcza macierz pomyłek na poziomie pacjenta (każdy pacjent oceniony
      dokładnie raz, jako element swojego foldu walidacyjnego).

Nie wymaga scikit-learn — grupowy k-fold jest deterministyczny (ten sam seed
co train_freq.py; stały podział przetasowanych pacjentów na k rozłącznych części).

Użycie:
    python kfold_freq.py --data data/processed/windows_td_rest.npz \
        --num-classes 2 --folds 5
"""

from __future__ import annotations

import argparse

import numpy as np

import tensorflow as tf
from tensorflow import keras

from train_freq import (
    build_feature_matrix,
    build_mlp,
    compute_class_weights,
    load_dataset,
    sanity_check_pd_band,
)


def group_kfold_subjects(subjects: np.ndarray, n_folds: int,
                         seed: int) -> list[set[str]]:
    """Dzieli unikalnych pacjentów na ``n_folds`` rozłącznych grup (deterministycznie)."""
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_subjects)
    return [set(fold.tolist()) for fold in np.array_split(shuffled, n_folds)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walidacja k-krotna grupowa (cechy częstotliwości).")
    parser.add_argument("--data", default="data/processed/windows_td_rest.npz")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    windows, labels, subjects, fs = load_dataset(args.data)
    valid_mask = labels >= 0
    windows, labels, subjects = windows[valid_mask], labels[valid_mask], subjects[valid_mask]

    n_channels = windows.shape[2]
    print(f"[kfold] Okien: {windows.shape[0]}, pacjentów: {len(set(subjects))}, "
          f"klasy: {np.unique(labels).tolist()}, fs={fs}")
    features = build_feature_matrix(windows, fs)

    # Model-niezależny test rozdzielności markera 4-6 Hz (raz, na całości).
    sanity_check_pd_band(features, labels, n_channels)

    folds = group_kfold_subjects(subjects, args.folds, args.seed)
    print(f"\n[kfold] Pacjentów w foldach: {[len(f) for f in folds]}")

    fold_balanced_acc: list[float] = []
    fold_pd_recall: list[float] = []
    # Zbiorcza macierz pomyłek na poziomie pacjenta (każdy pacjent raz).
    patient_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

    for fold_idx, val_subjects in enumerate(folds):
        val_mask = np.array([s in val_subjects for s in subjects])
        train_features, train_labels = features[~val_mask], labels[~val_mask]
        val_features, val_labels = features[val_mask], labels[val_mask]
        val_subject_ids = subjects[val_mask]

        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True) + 1e-8
        train_std = (train_features - mean) / std
        val_std = (val_features - mean) / std

        class_weight = (None if args.no_class_weights
                        else compute_class_weights(train_labels, args.num_classes))

        model = build_mlp(features.shape[1], args.num_classes)
        model.fit(
            train_std, train_labels,
            validation_data=(val_std, val_labels),
            epochs=args.epochs, batch_size=args.batch_size,
            class_weight=class_weight, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=12,
                restore_best_weights=True)],
        )

        probabilities = model.predict(val_std, verbose=0)

        # --- poziom okna ---
        window_pred = probabilities.argmax(axis=1)
        per_class_recall = []
        for c in range(args.num_classes):
            class_mask = val_labels == c
            recall = (window_pred[class_mask] == c).mean() if class_mask.any() else 0.0
            per_class_recall.append(float(recall))
        balanced_acc = float(np.mean(per_class_recall))
        fold_balanced_acc.append(balanced_acc)
        if args.num_classes >= 2:
            fold_pd_recall.append(per_class_recall[1])

        # --- poziom pacjenta (agregacja prawdopodobieństw) ---
        for subject in sorted(val_subjects):
            subj_mask = val_subject_ids == subject
            if not subj_mask.any():
                continue
            patient_pred = int(probabilities[subj_mask].mean(axis=0).argmax())
            patient_true = int(val_labels[subj_mask][0])
            patient_confusion[patient_true, patient_pred] += 1

        print(f"[kfold] fold {fold_idx + 1}/{args.folds}: "
              f"val pacjentów={len(val_subjects)}, balanced acc (okno)={balanced_acc:.3f}, "
              f"recall PD={per_class_recall[1] if args.num_classes >= 2 else float('nan'):.3f}")

    # --- Podsumowanie ---
    print("\n==================== PODSUMOWANIE k-fold ====================")
    print(f"[kfold] Balanced accuracy (okno): "
          f"{np.mean(fold_balanced_acc):.3f} ± {np.std(fold_balanced_acc):.3f}  "
          f"(foldy: {[round(a, 3) for a in fold_balanced_acc]})")
    if fold_pd_recall:
        print(f"[kfold] Recall PD (okno):         "
              f"{np.mean(fold_pd_recall):.3f} ± {np.std(fold_pd_recall):.3f}  "
              f"(foldy: {[round(a, 3) for a in fold_pd_recall]})")

    n_patients = patient_confusion.sum()
    patient_acc = np.trace(patient_confusion) / n_patients if n_patients else 0.0
    patient_recall = [
        patient_confusion[c, c] / patient_confusion[c, :].sum()
        if patient_confusion[c, :].sum() > 0 else 0.0
        for c in range(args.num_classes)
    ]
    print("\n[kfold] Zbiorcza macierz pomyłek (poziom PACJENTA, każdy pacjent raz):")
    print("        " + "".join(f"pred{c:<6}" for c in range(args.num_classes)))
    for c in range(args.num_classes):
        row = "".join(f"{patient_confusion[c, p]:<10}" for p in range(args.num_classes))
        print(f"  true{c}  {row}")
    print(f"\n[kfold] Trafność na poziomie PACJENTA: {patient_acc:.3f} "
          f"({int(np.trace(patient_confusion))}/{int(n_patients)})")
    print(f"[kfold] Balanced accuracy (pacjent):   {np.mean(patient_recall):.3f}  "
          f"(recall per klasa: {[round(r, 3) for r in patient_recall]})")


if __name__ == "__main__":
    main()
