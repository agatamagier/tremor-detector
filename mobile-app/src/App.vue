<script setup>
/**
 * App.vue
 * =======
 * Spina cały potok Edge AI:
 *   czujniki IMU -> filtr pasmowy 3-12 Hz (causalny) -> okno 3 s (50% overlap)
 *   -> standaryzacja -> model LSTM (TF.js) -> poziom drżenia -> adaptacja UI.
 *
 * Tryb demo: gdy model nie jest wczytany (brak public/model/), można ręcznie
 * ustawiać poziom nasilenia suwakiem, aby zobaczyć adaptację interfejsu.
 */
import { ref, onBeforeUnmount } from "vue";
import { StreamingBandpass, WindowBuffer } from "./utils/signal.js";
import { useTremorModel } from "./composables/useTremorModel.js";
import { useMotionSensors } from "./composables/useMotionSensors.js";
import { useAdaptiveUI } from "./composables/useAdaptiveUI.js";
import SurveyForm from "./components/SurveyForm.vue";
import TremorStatus from "./components/TremorStatus.vue";
import AdaptiveButton from "./components/AdaptiveButton.vue";

// --- Parametry sygnału (muszą odpowiadać preprocess.py) ---
const FS = 50;
const WINDOW_SEC = 3;
const WINDOW_SIZE = FS * WINDOW_SEC; // 150 próbek
const CHANNELS = 6;

const { isReady, isLoading, error, load, predict } = useTremorModel();
const sensors = useMotionSensors();
const { severity, profile, reportPrediction, setSeverity } = useAdaptiveUI();

const running = ref(false);
const lastProbs = ref([]);
const inferenceBusy = ref(false);

// Filtr + bufor okienkujący współdzielone przez sesję pomiarową.
let bandpass = null;
let windowBuffer = null;

async function startSession() {
  if (running.value) return;

  // Model jest opcjonalny — bez niego działa tryb demo (ręczny suwak).
  if (!isReady.value && !isLoading.value) {
    await load();
  }

  bandpass = new StreamingBandpass({
    fs: FS,
    lowcut: 3,
    highcut: 12,
    channels: CHANNELS,
    sections: 2,
  });

  windowBuffer = new WindowBuffer({
    windowSize: WINDOW_SIZE,
    channels: CHANNELS,
    overlap: 0.5,
    onWindow: handleWindow,
  });

  await sensors.start((sample) => {
    const filtered = bandpass.process(sample);
    windowBuffer.push(filtered);
  });

  running.value = true;
}

async function handleWindow(window) {
  if (!isReady.value || inferenceBusy.value) return;
  inferenceBusy.value = true;
  try {
    const result = await predict(window);
    if (result) {
      lastProbs.value = result.probabilities;
      reportPrediction(result.severity);
    }
  } finally {
    inferenceBusy.value = false;
  }
}

function stopSession() {
  sensors.stop();
  bandpass?.reset();
  windowBuffer?.reset();
  running.value = false;
}

function onSurveySubmit(answers) {
  console.log("[App] Ankieta zapisana:", answers);
  // TODO: zapis lokalny / wysyłka do backendu badawczego.
}

onBeforeUnmount(stopSession);
</script>

<template>
  <div class="app">
    <header>
      <h1>Tremor Adaptive UI</h1>
      <p class="sub">Edge AI · adaptacja interfejsu do nasilenia drżenia</p>
    </header>

    <TremorStatus
      :severity="severity"
      :label="profile.label"
      :probabilities="lastProbs"
      :is-ready="isReady"
    />

    <div class="controls">
      <AdaptiveButton
        v-if="!running"
        label="Start pomiaru"
        :tap-debounce-ms="profile.tapDebounceMs"
        :dead-zone="profile.deadZone"
        @press="startSession"
      />
      <AdaptiveButton
        v-else
        label="Stop"
        variant="secondary"
        :tap-debounce-ms="profile.tapDebounceMs"
        :dead-zone="profile.deadZone"
        @press="stopSession"
      />
    </div>

    <p v-if="isLoading" class="hint">Ładowanie modelu…</p>
    <p v-if="error" class="hint err">
      Model niewczytany ({{ String(error).slice(0, 80) }}). Działa tryb demo ↓
    </p>

    <!-- Tryb demo: ręczne sterowanie poziomem gdy brak modelu -->
    <div v-if="!isReady" class="demo">
      <label>Tryb demo — poziom drżenia: <strong>{{ severity }}</strong></label>
      <input
        type="range"
        min="0"
        max="3"
        step="1"
        :value="severity"
        @input="setSeverity(Number($event.target.value))"
      />
    </div>

    <SurveyForm
      :tap-debounce-ms="profile.tapDebounceMs"
      :dead-zone="profile.deadZone"
      @submit="onSurveySubmit"
    />

    <p v-if="sensors.permissionError.value" class="hint err">
      {{ sensors.permissionError.value }}
    </p>
  </div>
</template>

<style scoped>
.app {
  max-width: 480px;
  margin: 0 auto;
  padding: 20px 16px calc(20px + env(safe-area-inset-bottom));
  display: flex;
  flex-direction: column;
  gap: 16px;
}
header h1 {
  margin: 0;
  font-size: calc(1.5rem * var(--font-scale));
}
.sub {
  margin: 4px 0 0;
  color: #94a3b8;
  font-size: 0.85rem;
}
.controls {
  display: flex;
  gap: var(--touch-gap);
}
.hint {
  color: #94a3b8;
  font-size: 0.85rem;
  margin: 0;
}
.hint.err {
  color: #f59e0b;
}
.demo {
  background: #1e293b;
  border-radius: 14px;
  padding: 14px 16px;
}
.demo input {
  width: 100%;
  margin-top: 10px;
  height: var(--touch-target);
}
</style>
