<script setup>
/**
 * SurveyForm
 * ==========
 * Krótka ankieta samooceny (self-report) wypełniana przez pacjenta.
 * Opcje odpowiedzi używają AdaptiveButton, więc automatycznie powiększają się
 * przy wykrytym drżeniu — ankieta sama jest "testem" użyteczności adaptacji.
 */
import { reactive, computed } from "vue";
import AdaptiveButton from "./AdaptiveButton.vue";

const props = defineProps({
  tapDebounceMs: { type: Number, default: 200 },
  deadZone: { type: Number, default: 8 },
});

const emit = defineEmits(["submit"]);

const QUESTIONS = [
  { id: "rest_tremor", text: "Czy odczuwasz teraz drżenie w spoczynku?", options: ["Nie", "Lekko", "Wyraźnie", "Bardzo"] },
  { id: "medication", text: "Kiedy ostatnio przyjęto lek (ON/OFF)?", options: ["ON (po leku)", "OFF (przed)"] },
  { id: "difficulty", text: "Jak trudno trafić w przyciski?", options: ["Łatwo", "Średnio", "Trudno"] },
];

const answers = reactive({});
const allAnswered = computed(() =>
  QUESTIONS.every((q) => answers[q.id] !== undefined)
);

function choose(qid, optionIndex) {
  answers[qid] = optionIndex;
}

function submit() {
  if (!allAnswered.value) return;
  emit("submit", { ...answers });
}
</script>

<template>
  <div class="survey">
    <h2>Ankieta samooceny</h2>
    <div v-for="q in QUESTIONS" :key="q.id" class="question">
      <p class="q-text">{{ q.text }}</p>
      <div class="options">
        <AdaptiveButton
          v-for="(opt, i) in q.options"
          :key="i"
          :label="opt"
          :variant="answers[q.id] === i ? 'primary' : 'secondary'"
          :tap-debounce-ms="tapDebounceMs"
          :dead-zone="deadZone"
          @press="choose(q.id, i)"
        />
      </div>
    </div>

    <AdaptiveButton
      label="Zapisz ankietę"
      :tap-debounce-ms="tapDebounceMs"
      :dead-zone="deadZone"
      :style="{ opacity: allAnswered ? 1 : 0.5 }"
      @press="submit"
    />
  </div>
</template>

<style scoped>
.survey {
  background: #1e293b;
  border-radius: 14px;
  padding: 16px;
}
h2 {
  margin: 0 0 12px;
  font-size: calc(1.2rem * var(--font-scale));
}
.question {
  margin-bottom: 18px;
}
.q-text {
  margin: 0 0 8px;
  font-size: calc(1rem * var(--font-scale));
}
.options {
  display: flex;
  flex-wrap: wrap;
  gap: var(--touch-gap);
}
</style>
