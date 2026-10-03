/**
 * useAdaptiveUI.js
 * ================
 * Tłumaczy przewidziany poziom nasilenia drżenia (0-3) na konkretne parametry
 * interfejsu i zapisuje je do zmiennych CSS (:root) oraz udostępnia reaktywnie.
 *
 * Mapowanie celów dotykowych: 48dp (brak/łagodne) -> 72dp (silne) — zgodnie
 * z zaleceniami dostępności (min. 48dp) z zapasem dla pacjentów z drżeniem.
 *
 * Histereza: zmiana poziomu wymaga potwierdzenia przez kilka kolejnych
 * predykcji, aby UI nie "migotało" między stanami przy granicznych wynikach.
 */

import { ref, computed } from "vue";

/** Profile UI dla kolejnych poziomów nasilenia (indeks = klasa). */
export const SEVERITY_PROFILES = [
  { label: "Brak / minimalne", touchTarget: 48, gap: 12, fontScale: 1.0, emaAlpha: 0.6, deadZone: 4, tapDebounceMs: 150 },
  { label: "Łagodne", touchTarget: 56, gap: 16, fontScale: 1.1, emaAlpha: 0.45, deadZone: 8, tapDebounceMs: 220 },
  { label: "Umiarkowane", touchTarget: 64, gap: 20, fontScale: 1.2, emaAlpha: 0.3, deadZone: 14, tapDebounceMs: 300 },
  { label: "Silne", touchTarget: 72, gap: 28, fontScale: 1.35, emaAlpha: 0.2, deadZone: 20, tapDebounceMs: 400 },
];

export function useAdaptiveUI({ hysteresis = 3 } = {}) {
  const severity = ref(0); // zatwierdzony, stabilny poziom
  const profile = computed(() => SEVERITY_PROFILES[severity.value]);

  let candidate = 0;
  let candidateCount = 0;

  /** Zastosuj parametry profilu do zmiennych CSS. */
  function applyToCss(p) {
    const root = document.documentElement.style;
    root.setProperty("--touch-target", `${p.touchTarget}px`);
    root.setProperty("--touch-gap", `${p.gap}px`);
    root.setProperty("--font-scale", String(p.fontScale));
  }

  /**
   * Zgłoś nowy wynik predykcji. Zmiana poziomu następuje dopiero po
   * `hysteresis` kolejnych zgodnych predykcjach.
   * @param {number} predicted klasa z modelu (0-3)
   */
  function reportPrediction(predicted) {
    if (predicted === severity.value) {
      candidate = predicted;
      candidateCount = 0;
      return;
    }
    if (predicted === candidate) {
      candidateCount++;
    } else {
      candidate = predicted;
      candidateCount = 1;
    }
    if (candidateCount >= hysteresis) {
      severity.value = candidate;
      candidateCount = 0;
      applyToCss(SEVERITY_PROFILES[severity.value]);
    }
  }

  /** Ręczne ustawienie poziomu (np. z ankiety / trybu demo). */
  function setSeverity(level) {
    severity.value = Math.min(SEVERITY_PROFILES.length - 1, Math.max(0, level));
    candidate = severity.value;
    candidateCount = 0;
    applyToCss(profile.value);
  }

  return { severity, profile, reportPrediction, setSeverity };
}
