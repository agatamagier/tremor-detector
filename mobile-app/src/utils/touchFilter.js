/**
 * touchFilter.js
 * ==============
 * Filtrowanie "jitteru" zdarzeń dotykowych wywołanego drżeniem dłoni.
 *
 * Dwa mechanizmy:
 *   1. EMA (wykładnicze wygładzanie) pozycji wskaźnika — tłumi mikrodrgania
 *      podczas przesuwania palca.
 *   2. Dead-zone na starcie dotyku — ignoruje niezamierzone mikroruchy.
 *   3. Anty-dubel (debounce) dla "tapów" — drżenie potrafi wygenerować kilka
 *      szybkich zdarzeń pointerdown; akceptujemy tylko pierwsze w oknie czasu.
 *
 * Siła filtrowania skaluje się z nasileniem drżenia (parametry z useAdaptiveUI).
 */

/**
 * Wygładzanie pozycji wskaźnika metodą EMA.
 * alpha bliskie 1 = słabe wygładzanie; bliskie 0 = mocne (większe opóźnienie).
 */
export class PointerSmoother {
  /** @param {number} alpha współczynnik EMA (0-1) */
  constructor(alpha = 0.4) {
    this.alpha = alpha;
    this.x = null;
    this.y = null;
  }

  setAlpha(alpha) {
    this.alpha = Math.min(1, Math.max(0.05, alpha));
  }

  /** Zwraca wygładzoną pozycję { x, y }. */
  update(x, y) {
    if (this.x === null) {
      this.x = x;
      this.y = y;
    } else {
      this.x = this.alpha * x + (1 - this.alpha) * this.x;
      this.y = this.alpha * y + (1 - this.alpha) * this.y;
    }
    return { x: this.x, y: this.y };
  }

  reset() {
    this.x = null;
    this.y = null;
  }
}

/**
 * Debounce "tapów" — chroni przed podwójnym wyzwoleniem przez drżenie.
 * Zwraca true, jeśli zdarzenie powinno zostać ZAAKCEPTOWANE.
 *
 * @param {object} opts
 * @param {number} opts.windowMs minimalny odstęp między akceptowanymi tapami
 */
export function createTapDebounce({ windowMs = 300 } = {}) {
  let lastAcceptedAt = 0;
  let windowSize = windowMs;

  return {
    setWindow(ms) {
      windowSize = ms;
    },
    /** @returns {boolean} czy zaakceptować tap */
    accept() {
      const now = performance.now();
      if (now - lastAcceptedAt >= windowSize) {
        lastAcceptedAt = now;
        return true;
      }
      return false;
    },
    reset() {
      lastAcceptedAt = 0;
    },
  };
}

/**
 * Sprawdza, czy ruch mieści się w "strefie martwej" (uznawany za drżenie,
 * nie za intencjonalny gest). Dystans w px względem punktu startowego dotyku.
 *
 * @param {number} dx
 * @param {number} dy
 * @param {number} radius promień strefy martwej [px]
 * @returns {boolean} true jeśli ruch jest poniżej progu (do zignorowania)
 */
export function isWithinDeadZone(dx, dy, radius) {
  return Math.hypot(dx, dy) < radius;
}
