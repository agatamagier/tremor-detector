"""
eval_patient_level.py
=====================
Ewaluacja modelu cech częstotliwościowych na poziomie PACJENTA, a nie
pojedynczego okna 3 s. Klinicznie diagnozuje się pacjenta — więc agregujemy
predykcje modelu po wszystkich oknach danego pacjenta (średnie prawdopodobieństwo
-> jedna diagnoza na pacjenta) i liczymy trafność pacjent-po-pacjencie.

Używa tego samego, deterministycznego splitu per-pacjent co train_freq.py
(ten sam seed i val-split), żeby oceniać na pacjentach NIEwidzianych w treningu.

Użycie:
    python eval_patient_level.py --data data/processed/windows_2class.npz \
        --model models_freq_2class --num-classes 2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tensorflow import keras

from train_freq import load_dataset, build_feature_matrix


def reproduce_val_subjects(subjects: np.ndarray, seed: int,
                           val_split: float) -> set[str]:
    """Odtwarza DOKŁADNIE ten sam zbiór pacjentów walidacyjnych co train_freq.py."""
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_subjects)
    n_val = max(1, int(len(shuffled) * val_split))
    return set(shuffled[:n_val].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja per-pacjent (agregacja okien).")
    parser.add_argument("--data", default="data/processed/windows_2class.npz")
    parser.add_argument("--model", default="models_freq_2class")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    windows, labels, subjects, fs = load_dataset(args.data)
    valid_mask = labels >= 0
    windows, labels, subjects = (windows[valid_mask], labels[valid_mask],
                                 subjects[valid_mask])

    features = build_feature_matrix(windows, fs)

    # Standaryzacja tymi samymi statystykami, co przy treningu.
    norm = np.load(os.path.join(args.model, "feature_norm.npz"))
    features = (features - norm["mean"]) / (norm["std"] + 0.0)

    val_subjects = reproduce_val_subjects(subjects, args.seed, args.val_split)
    val_mask = np.array([s in val_subjects for s in subjects])

    model = keras.models.load_model(os.path.join(args.model, "tremor_freq_mlp.keras"))
    probabilities = model.predict(features[val_mask], verbose=0)
    val_subject_ids = subjects[val_mask]
    val_labels = labels[val_mask]

    # --- Poziom OKNA (dla porównania) ---
    window_pred = probabilities.argmax(axis=1)
    window_acc = (window_pred == val_labels).mean()

    # --- Poziom PACJENTA: uśrednij prawdopodobieństwa po oknach pacjenta ---
    print(f"[patient] Pacjenci walidacyjni: {len(val_subjects)}\n")
    print("  pacjent   prawda  pred  #okien  p(PD)")
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for subject in sorted(val_subjects):
        subj_mask = val_subject_ids == subject
        if not subj_mask.any():
            continue
        mean_prob = probabilities[subj_mask].mean(axis=0)
        patient_pred = int(mean_prob.argmax())
        patient_true = int(val_labels[subj_mask][0])  # wszystkie okna pacjenta mają tę samą etykietę
        confusion[patient_true, patient_pred] += 1
        p_pd = mean_prob[1] if args.num_classes >= 2 else float("nan")
        flag = "" if patient_pred == patient_true else "  <-- blad"
        print(f"  {subject:<9} {patient_true:<7} {patient_pred:<5} "
              f"{int(subj_mask.sum()):<7} {p_pd:.2f}{flag}")

    n_patients = confusion.sum()
    patient_acc = np.trace(confusion) / n_patients if n_patients else 0.0
    per_class_recall = [
        confusion[c, c] / confusion[c, :].sum() if confusion[c, :].sum() > 0 else 0.0
        for c in range(args.num_classes)
    ]

    print("\n[patient] Macierz pomyłek (poziom pacjenta):")
    print("        " + "".join(f"pred{c:<6}" for c in range(args.num_classes)))
    for c in range(args.num_classes):
        row = "".join(f"{confusion[c, p]:<10}" for p in range(args.num_classes))
        print(f"  true{c}  {row}")

    print(f"\n[patient] Trafność na poziomie OKNA:    {window_acc:.3f}")
    print(f"[patient] Trafność na poziomie PACJENTA: {patient_acc:.3f} "
          f"({int(np.trace(confusion))}/{int(n_patients)})")
    print(f"[patient] Balanced accuracy (pacjent):   {np.mean(per_class_recall):.3f}")


if __name__ == "__main__":
    main()
