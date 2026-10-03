<script setup>
/**
 * TremorStatus
 * ============
 * Wizualizacja aktualnego, zatwierdzonego poziomu nasilenia drżenia
 * oraz surowych prawdopodobieństw z ostatniej predykcji.
 */
defineProps({
  severity: { type: Number, default: 0 },
  label: { type: String, default: "—" },
  probabilities: { type: Array, default: () => [] },
  isReady: { type: Boolean, default: false },
});

const COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"];
</script>

<template>
  <div class="status">
    <div class="row">
      <span class="dot" :style="{ background: COLORS[severity] || '#64748b' }" />
      <span class="label">Drżenie: <strong>{{ label }}</strong></span>
      <span class="ready" :class="{ on: isReady }">
        {{ isReady ? "model gotowy" : "model nieaktywny" }}
      </span>
    </div>
    <div v-if="probabilities.length" class="bars">
      <div v-for="(p, i) in probabilities" :key="i" class="bar-wrap">
        <div
          class="bar"
          :style="{ height: `${Math.round(p * 100)}%`, background: COLORS[i] || '#64748b' }"
        />
        <span class="bar-label">{{ i }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.status {
  background: #1e293b;
  border-radius: 14px;
  padding: 14px 16px;
}
.row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
}
.label {
  flex: 1;
}
.ready {
  font-size: 0.75rem;
  color: #94a3b8;
}
.ready.on {
  color: #22c55e;
}
.bars {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  height: 64px;
  margin-top: 12px;
}
.bar-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  justify-content: flex-end;
}
.bar {
  width: 100%;
  border-radius: 4px 4px 0 0;
  min-height: 2px;
  transition: height 0.2s ease;
}
.bar-label {
  font-size: 0.7rem;
  color: #94a3b8;
  margin-top: 4px;
}
</style>
