<script setup>
/**
 * AdaptiveButton
 * ==============
 * Przycisk, którego rozmiar sterowany jest zmienną CSS --touch-target
 * (48dp -> 72dp wg nasilenia drżenia). Chroni przed przypadkowym
 * wielokrotnym wyzwoleniem (debounce) oraz "ślizganiem" palca po celu
 * (dead-zone liczony od punktu rozpoczęcia dotyku).
 */
import { onBeforeUnmount } from "vue";
import { createTapDebounce, isWithinDeadZone } from "../utils/touchFilter.js";

const props = defineProps({
  label: { type: String, default: "" },
  /** Siła filtrowania dotyku — zwykle z useAdaptiveUI().profile */
  tapDebounceMs: { type: Number, default: 200 },
  deadZone: { type: Number, default: 8 },
  variant: { type: String, default: "primary" }, // primary | secondary
});

const emit = defineEmits(["press"]);

const debounce = createTapDebounce({ windowMs: props.tapDebounceMs });
let startX = 0;
let startY = 0;
let moved = false;

function onPointerDown(e) {
  startX = e.clientX;
  startY = e.clientY;
  moved = false;
}

function onPointerMove(e) {
  // Jeśli palec wyszedł poza strefę martwą -> traktuj jako gest (nie tap).
  if (!isWithinDeadZone(e.clientX - startX, e.clientY - startY, props.deadZone)) {
    moved = true;
  }
}

function onPointerUp() {
  debounce.setWindow(props.tapDebounceMs);
  if (moved) return; // ruch poza strefą -> ignoruj (prawdopodobnie drżenie)
  if (!debounce.accept()) return; // zbyt szybki kolejny tap -> ignoruj
  emit("press");
}

onBeforeUnmount(() => debounce.reset());
</script>

<template>
  <button
    class="adaptive-btn"
    :class="variant"
    @pointerdown="onPointerDown"
    @pointermove="onPointerMove"
    @pointerup="onPointerUp"
  >
    <slot>{{ label }}</slot>
  </button>
</template>

<style scoped>
.adaptive-btn {
  min-height: var(--touch-target);
  min-width: var(--touch-target);
  padding: 0 calc(var(--touch-gap) * 1.2);
  margin: calc(var(--touch-gap) / 2);
  font-size: calc(1rem * var(--font-scale));
  border: none;
  border-radius: 14px;
  font-weight: 600;
  color: #fff;
  touch-action: manipulation; /* wyłącza zoom na double-tap */
  transition: min-height 0.2s ease, min-width 0.2s ease, font-size 0.2s ease;
}
.adaptive-btn.primary {
  background: #2563eb;
}
.adaptive-btn.secondary {
  background: #475569;
}
.adaptive-btn:active {
  filter: brightness(0.85);
}
</style>
