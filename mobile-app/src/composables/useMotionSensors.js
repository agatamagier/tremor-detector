/**
 * useMotionSensors.js
 * ===================
 * Dostęp do akcelerometru i żyroskopu z jednolitym interfejsem:
 *   - natywnie przez @capacitor/motion (Android/iOS w Capacitorze),
 *   - fallback przez Web API `devicemotion` (przeglądarka / dev w Vite).
 *
 * Emituje ramki w kolejności kanałów zgodnej z preprocess.py:
 *   [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
 * (gyro = rotationRate; na webie w deg/s — jednostki muszą być spójne z danymi
 *  treningowymi, patrz uwaga niżej).
 */

import { ref } from "vue";
import { Capacitor } from "@capacitor/core";

export function useMotionSensors() {
  const isListening = ref(false);
  const permissionError = ref(null);

  let removeListener = null;
  let onSample = null;

  function emit(accel, rotation) {
    if (!onSample) return;
    onSample([
      accel?.x ?? 0,
      accel?.y ?? 0,
      accel?.z ?? 0,
      rotation?.alpha ?? 0,
      rotation?.beta ?? 0,
      rotation?.gamma ?? 0,
    ]);
  }

  /**
   * @param {(sample:number[]) => void} callback wywoływany dla każdej ramki
   */
  async function start(callback) {
    if (isListening.value) return;
    onSample = callback;
    permissionError.value = null;

    if (Capacitor.isNativePlatform()) {
      // Ścieżka natywna (Capacitor).
      const { Motion } = await import("@capacitor/motion");
      const handle = await Motion.addListener("accel", (event) => {
        // event: { acceleration, accelerationIncludingGravity, rotationRate, interval }
        emit(event.acceleration, event.rotationRate);
      });
      removeListener = () => handle.remove();
      isListening.value = true;
    } else {
      // Ścieżka webowa — wymaga HTTPS; iOS wymaga zgody użytkownika.
      if (
        typeof DeviceMotionEvent !== "undefined" &&
        typeof DeviceMotionEvent.requestPermission === "function"
      ) {
        try {
          const state = await DeviceMotionEvent.requestPermission();
          if (state !== "granted") {
            permissionError.value = "Brak zgody na dostęp do czujników ruchu.";
            return;
          }
        } catch (e) {
          permissionError.value = String(e);
          return;
        }
      }

      const handler = (event) => {
        emit(event.acceleration, event.rotationRate);
      };
      window.addEventListener("devicemotion", handler);
      removeListener = () =>
        window.removeEventListener("devicemotion", handler);
      isListening.value = true;
    }
  }

  function stop() {
    if (removeListener) {
      removeListener();
      removeListener = null;
    }
    onSample = null;
    isListening.value = false;
  }

  return { isListening, permissionError, start, stop };
}
