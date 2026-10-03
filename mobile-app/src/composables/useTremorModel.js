/**
 * useTremorModel.js
 * =================
 * Ładowanie modelu LSTM w formacie TensorFlow.js z lokalnego pliku .json
 * oraz inferencja na pojedynczym oknie sygnału.
 *
 * Oczekiwane assety (skopiuj z ml-pipeline/models/ do public/model/):
 *   - public/model/model.json           (+ pliki *.bin z wagami)
 *   - public/model/normalization.json   { mean: [...], std: [...], channels: [...] }
 *
 * normalization.json NIE jest produkowany domyślnie przez train.py (zapisuje
 * .npz) — patrz public/model/README.md: wystarczy jednorazowo przekonwertować
 * normalization.npz -> JSON.
 */

import { ref, shallowRef } from "vue";
import * as tf from "@tensorflow/tfjs";
import { standardizeWindow } from "../utils/signal.js";

export function useTremorModel(options = {}) {
  const {
    modelUrl = "/model/model.json",
    normUrl = "/model/normalization.json",
  } = options;

  const model = shallowRef(null);
  const normalization = ref(null); // { mean, std, channels }
  const isLoading = ref(false);
  const isReady = ref(false);
  const error = ref(null);

  async function load() {
    if (isReady.value || isLoading.value) return;
    isLoading.value = true;
    error.value = null;
    try {
      model.value = await tf.loadLayersModel(modelUrl);

      // Statystyki normalizacji (opcjonalne — bez nich pomijamy z-score).
      try {
        const res = await fetch(normUrl);
        if (res.ok) {
          normalization.value = await res.json();
        } else {
          console.warn(
            "[useTremorModel] Brak normalization.json — inferencja bez z-score."
          );
        }
      } catch {
        console.warn("[useTremorModel] Nie wczytano normalization.json.");
      }

      // "Rozgrzewka" — pierwsza predykcja alokuje bufory GPU/WASM.
      const inputShape = model.value.inputs[0].shape; // [null, time, channels]
      const warm = tf.zeros([1, inputShape[1], inputShape[2]]);
      const out = model.value.predict(warm);
      await out.data();
      tf.dispose([warm, out]);

      isReady.value = true;
    } catch (e) {
      error.value = e;
      console.error("[useTremorModel] Błąd ładowania modelu:", e);
    } finally {
      isLoading.value = false;
    }
  }

  /**
   * Inferencja na jednym oknie.
   * @param {number[][]} window (windowSize x channels)
   * @returns {Promise<{ severity: number, probabilities: number[] }|null>}
   */
  async function predict(window) {
    if (!isReady.value || !model.value) return null;

    let input = window;
    if (normalization.value) {
      input = standardizeWindow(
        window,
        normalization.value.mean,
        normalization.value.std
      );
    }

    return tf.tidy(() => {
      const x = tf.tensor3d([input]); // [1, time, channels]
      const logits = model.value.predict(x);
      const probs = logits.dataSync();
      const severity = probs.indexOf(Math.max(...probs));
      return { severity, probabilities: Array.from(probs) };
    });
  }

  function dispose() {
    if (model.value) {
      model.value.dispose();
      model.value = null;
    }
    isReady.value = false;
  }

  return { model, normalization, isLoading, isReady, error, load, predict, dispose };
}
