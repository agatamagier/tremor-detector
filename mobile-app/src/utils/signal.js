/**
 * signal.js
 * =========
 * Przetwarzanie sygnału IMU w czasie rzeczywistym po stronie urządzenia —
 * mobilny odpowiednik `ml-pipeline/preprocess.py`.
 *
 * RÓŻNICA WZGLĘDEM PIPELINE'U OFFLINE:
 *   preprocess.py używa `scipy.signal.filtfilt` (filtr dwukierunkowy,
 *   zero-phase) — to wymaga znajomości CAŁEGO sygnału, więc nie da się go
 *   zastosować strumieniowo. Na urządzeniu filtrujemy CAUSALNIE (strumieniowo),
 *   utrzymując stan filtra między próbkami.
 *
 *   Dla ścisłej parzystości pasma zalecane jest WYEKSPORTOWANIE współczynników
 *   (b, a) z Pythona do JSON i wczytanie ich tutaj (patrz `loadCoeffsFromJson`).
 *   Domyślnie poniżej projektujemy biquadowy filtr pasmowy (RBJ cookbook)
 *   jako samowystarczalny fallback dla pasma drżenia 3-12 Hz.
 */

/**
 * Projektuje pojedynczą sekcję biquad (bandpass, constant 0 dB peak gain)
 * metodą RBJ Audio EQ Cookbook. Zwraca znormalizowane współczynniki
 * { b: [b0,b1,b2], a: [1,a1,a2] }.
 *
 * @param {number} fs      częstotliwość próbkowania [Hz]
 * @param {number} lowcut  dolna granica pasma [Hz]
 * @param {number} highcut górna granica pasma [Hz]
 */
export function designBandpassBiquad(fs, lowcut, highcut) {
  const f0 = Math.sqrt(lowcut * highcut); // częstotliwość środkowa (geom.)
  const bw = highcut - lowcut;
  const w0 = (2 * Math.PI * f0) / fs;
  const cosw0 = Math.cos(w0);
  const sinw0 = Math.sin(w0);
  // Q z szerokości pasma względem częstotliwości środkowej.
  const Q = f0 / bw;
  const alpha = sinw0 / (2 * Q);

  const b0 = alpha;
  const b1 = 0;
  const b2 = -alpha;
  const a0 = 1 + alpha;
  const a1 = -2 * cosw0;
  const a2 = 1 - alpha;

  return {
    b: [b0 / a0, b1 / a0, b2 / a0],
    a: [1, a1 / a0, a2 / a0],
  };
}

/**
 * Strumieniowy filtr pasmowy dla WIELU kanałów jednocześnie.
 * Kaskaduje `sections` identycznych biquadów dla stromszego zbocza
 * (odpowiednik wyższego rzędu). Utrzymuje stan (Direct Form II Transposed)
 * osobno dla każdego kanału i każdej sekcji.
 */
export class StreamingBandpass {
  /**
   * @param {object} opts
   * @param {number} opts.fs
   * @param {number} opts.lowcut
   * @param {number} opts.highcut
   * @param {number} opts.channels liczba kanałów (np. 6: accel xyz + gyro xyz)
   * @param {number} opts.sections liczba kaskadowanych biquadów (domyślnie 2)
   */
  constructor({ fs, lowcut, highcut, channels, sections = 2 }) {
    this.channels = channels;
    this.sections = sections;
    const coeff = designBandpassBiquad(fs, lowcut, highcut);
    this.b = coeff.b;
    this.a = coeff.a;
    // Stan: [kanał][sekcja] -> { z1, z2 }
    this.state = Array.from({ length: channels }, () =>
      Array.from({ length: sections }, () => ({ z1: 0, z2: 0 }))
    );
  }

  /** Przetwarza jedną ramkę (tablica długości `channels`). Zwraca nową tablicę. */
  process(sample) {
    const out = new Array(this.channels);
    for (let ch = 0; ch < this.channels; ch++) {
      let x = sample[ch];
      for (let s = 0; s < this.sections; s++) {
        const st = this.state[ch][s];
        // Direct Form II Transposed
        const y = this.b[0] * x + st.z1;
        st.z1 = this.b[1] * x - this.a[1] * y + st.z2;
        st.z2 = this.b[2] * x - this.a[2] * y;
        x = y;
      }
      out[ch] = x;
    }
    return out;
  }

  reset() {
    for (const chState of this.state) {
      for (const st of chState) {
        st.z1 = 0;
        st.z2 = 0;
      }
    }
  }
}

/**
 * Bufor okienkujący (sliding window) — odpowiednik `sliding_windows()`.
 * Zbiera ramki aż do zapełnienia okna, po czym wywołuje callback z oknem
 * i przesuwa się o `hop` próbek (overlap = 1 - hop/windowSize).
 */
export class WindowBuffer {
  /**
   * @param {object} opts
   * @param {number} opts.windowSize liczba próbek w oknie (np. 150 dla 3 s @ 50 Hz)
   * @param {number} opts.channels
   * @param {number} opts.overlap nakładanie (0-1), domyślnie 0.5
   * @param {(window: number[][]) => void} opts.onWindow callback z gotowym oknem
   */
  constructor({ windowSize, channels, overlap = 0.5, onWindow }) {
    this.windowSize = windowSize;
    this.channels = channels;
    this.hop = Math.max(1, Math.round(windowSize * (1 - overlap)));
    this.onWindow = onWindow;
    this.buffer = []; // tablica ramek [ [ch0..chN], ... ]
    this.sinceLastEmit = 0;
  }

  push(sample) {
    this.buffer.push(sample);
    if (this.buffer.length > this.windowSize) {
      this.buffer.shift(); // utrzymujemy co najwyżej windowSize ramek
    }
    this.sinceLastEmit++;

    if (
      this.buffer.length === this.windowSize &&
      this.sinceLastEmit >= this.hop
    ) {
      this.sinceLastEmit = 0;
      // kopia, by callback nie trzymał referencji do żywego bufora
      this.onWindow(this.buffer.map((frame) => frame.slice()));
    }
  }

  reset() {
    this.buffer = [];
    this.sinceLastEmit = 0;
  }
}

/**
 * Standaryzacja z-score okna przy użyciu statystyk z treningu
 * (normalization.json wyeksportowany z train.py).
 *
 * @param {number[][]} window  (windowSize x channels)
 * @param {number[]} mean      długości channels
 * @param {number[]} std       długości channels
 * @returns {number[][]} znormalizowane okno
 */
export function standardizeWindow(window, mean, std) {
  return window.map((frame) =>
    frame.map((v, ch) => (v - mean[ch]) / (std[ch] + 1e-8))
  );
}
