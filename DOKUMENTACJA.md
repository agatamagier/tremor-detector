# Dokumentacja projektu Tremor Detector

> Dokument statusowy — zweryfikowany stan repozytorium na dzień **2026-10-19**.
> Każde stwierdzenie poniżej zostało potwierdzone bezpośrednio w kodzie / plikach,
> a nie przepisane z wcześniejszych notatek.

---

## 1. Czym jest projekt

Projekt badawczy *"Adaptive Mobile Interface Responding to Hand Tremor Severity in
Patients with Parkinson's Disease"*. Monorepo z dwiema niezależnymi częściami:

| Część | Technologie | Rola |
|---|---|---|
| **`ml-pipeline/`** | Python 3.11, TensorFlow/Keras, SciPy, pandas | Offline: pobranie PADS → filtr Butterwortha 3–12 Hz → okienkowanie → trening LSTM → (eksport do TF.js) |
| **`mobile-app/`** | Vue 3, Vite, Tailwind, Capacitor, TensorFlow.js | Aplikacja mobilna: ankieta medyczna z adaptacją UI + (docelowo) inferencja modelu na urządzeniu |

**Ważne:** projekt rozwinął się w **dwa równoległe tory**, które nie są jeszcze
ze sobą połączone (patrz §4):

- **Tor A — Edge AI** (pierwotny): czujniki IMU → filtr → okno → LSTM → poziom
  drżenia → adaptacja UI. Kod jest kompletny, ale **odpięty** od głównego widoku.
- **Tor B — Badanie HCI** (bieżąca praca): ankieta medyczna w dwóch trybach UI
  (standard vs adapted) z pomiarem metryk (czas wykonania, błędne kliknięcia).
  To jest to, co obecnie uruchamia `App.vue`.

---

## 2. Stan ogólny — na jakim etapie jesteśmy

**Faza: działający prototyp end-to-end na małej próbce danych + pivot w stronę badania HCI.**

- ✅ **ml-pipeline**: cały potok (pobieranie → konwersja → preprocessing → trening)
  działa i został **faktycznie przepuszczony** na próbce 5 pacjentów. Istnieje
  wytrenowany model Keras.
- ✅ **mobile-app**: buduje się (`dist/` istnieje), zależności zainstalowane,
  platforma Android dodana (`android/` istnieje). Ankieta HCI jest w pełni
  funkcjonalna w trybie standard/adapted.
- ⚠️ **Integracja ML ↔ aplikacja: NIE istnieje.** Model nie został wyeksportowany
  do TensorFlow.js ani skopiowany do aplikacji. Potok Edge AI w aplikacji nie
  jest podpięty do widoku.
- ⚠️ **Dane: tylko próbka testowa** (5 pacjentów) — model to proof-of-concept,
  nie model o wartości klinicznej.

---

## 3. Szczegółowy stan komponentów

### 3.1. `ml-pipeline/` — potok ML (Python)

| Plik | Stan | Opis |
|---|---|---|
| `download_pads.py` | ✅ kompletny | 4 tryby CLI: `download` (cały ZIP), `fetch` (celowane pobieranie wybranych zadań/pacjentów z manifestu SHA256SUMS), `inspect` (weryfikacja formatu), `convert` (PADS → nasz CSV + `labels.csv`). Mapowanie diagnozy: Healthy→0, Parkinson's→1, inne→2. Pobieranie atomowe (`.part` + `os.replace`), wznawialne. |
| `preprocess.py` | ✅ kompletny | `load_csv` → `resample_uniform` → `apply_bandpass` (Butterworth 3–12 Hz, `filtfilt` zero-phase) → `sliding_windows` (okno 3 s, 50% overlap). Dołącza etykiety i `subject_id` z `labels.csv`. Zapis do `windows.npz`. |
| `train.py` | ✅ kompletny | LSTM (64→32) + Dense, softmax. Split **per-pacjent** (bez wycieku okien), standaryzacja z-score liczona tylko na train, **balansowanie klas** (class weights), ewaluacja per-klasa (macierz pomyłek, precision/recall/F1, balanced accuracy), eksport Keras + `normalization.json/.npz`, próba eksportu TF.js. |
| `requirements.txt` | ✅ | numpy, pandas, scipy, tensorflow (rdzeń, bez tensorflowjs). |
| `requirements-tfjs.txt` | ✅ | tensorflowjs — osobny venv (konflikt `uvloop` na Windows). |

**Artefakty wygenerowane** (w `ml-pipeline/models/`):
- ✅ `tremor_lstm.keras` (423 KB) — wytrenowany model.
- ✅ `normalization.json` + `normalization.npz` — statystyki z-score (6 kanałów).
- ❌ **brak `tfjs_model/`** — eksport do TensorFlow.js nie został wykonany.

### 3.2. `mobile-app/` — aplikacja (Vue 3)

**Tor B (aktywny, uruchamiany przez `App.vue`):**

| Plik | Stan | Opis |
|---|---|---|
| `App.vue` | ✅ aktywny | Renderuje wyłącznie `MedicalSurvey`. Handlery `onSurveySubmit`/`onModeChange` to na razie `console.log` (TODO: zapis/eksport danych). |
| `components/MedicalSurvey.vue` | ✅ kompletny, **nietrackowany w git** | „Ankieta Samopoczucia i Leków". 2 tryby UI (standard 48 dp / adapted 72 dp), panel badacza z przełącznikiem trybu i metrykami na żywo, modal podsumowania. 7 pól (ID, pora dnia, lek, symptomy, suwak drżenia, toggle po posiłku). Debounce podwójnych tapnięć w trybie adapted. |
| `composables/useUxMetrics.js` | ✅ kompletny, **nietrackowany w git** | Metryki HCI: TCT (stoper co 100 ms) + missclicks (kliknięcia w tło poza elementem interaktywnym). |
| `utils/touchFilter.js` | ✅ | `PointerSmoother` (EMA), `createTapDebounce`, `isWithinDeadZone`. Używany przez `MedicalSurvey` (tap debounce). |

**Tor A (Edge AI — kompletny kod, ale ODPIĘTY od `App.vue`):**

| Plik | Stan | Opis |
|---|---|---|
| `composables/useTremorModel.js` | ✅ kod gotowy, ⚠️ nieużywany | Ładuje model TF.js z `/model/model.json` + `normalization.json`, warm-up, `predict(window)` → `{severity, probabilities}`. **Bez plików modelu działa tylko tryb demo.** |
| `composables/useMotionSensors.js` | ✅ kod gotowy, ⚠️ nieużywany | Akcelerometr + żyroskop przez `@capacitor/motion` (natywnie) lub `devicemotion` (web). Emituje `[accel_xyz, gyro_xyz]`. |
| `composables/useAdaptiveUI.js` | ✅ kod gotowy, ⚠️ nieużywany | Mapuje poziom 0–3 na profil UI (touch target 48→72 px, odstępy, font, filtry) z histerezą. |
| `utils/signal.js` | ✅ kod gotowy, ⚠️ nieużywany | `StreamingBandpass` (biquad RBJ, filtr **causalny** strumieniowy), `WindowBuffer`, `standardizeWindow`. Mobilny odpowiednik `preprocess.py`. |
| `components/AdaptiveButton.vue`, `SurveyForm.vue`, `TremorStatus.vue` | ⚠️ osierocone | Istnieją, ale `App.vue` ich już nie importuje (pozostałość po pierwotnym widoku Edge AI). |

**Konfiguracja / build:**
- ✅ `node_modules/` zainstalowane (201 pakietów), `dist/` zbudowane, `android/` dodane (`cap add android`).
- ✅ `public/model/README.md` — instrukcja podłączenia modelu. Katalog modelu **pusty** (tylko README).

---

## 4. Architektura i przepływ danych

### Tor A — Edge AI (zaprojektowany, niezłożony end-to-end)

```
  OFFLINE (ml-pipeline, Python)
  PADS → download_pads.py → preprocess.py → train.py → tremor_lstm.keras
         (PADS→CSV+labels)   (3–12 Hz, okna 3 s)   (LSTM 3 klasy)      │
                                                                       ▼
                                           [BRAKUJĄCY KROK: eksport TF.js
                                            + kopia do public/model/]
                                                                       │
  ON-DEVICE (mobile-app, Vue + TF.js)                                  ▼
  czujniki IMU → StreamingBandpass → WindowBuffer → useTremorModel → klasa
  (useMotionSensors)  (signal.js, causalny)  (okno)   (predict)        │
                                                        useAdaptiveUI (histereza)
                                                                       │
                                            zmienne CSS 48→72 px + touchFilter
```

### Tor B — Badanie HCI (działa dziś)

```
  App.vue → MedicalSurvey.vue
              ├── tryb UI: standard (48 dp) / adapted (72 dp)   ← przełączany ręcznie w panelu badacza
              ├── useUxMetrics: TCT + missclicks                ← pomiar na żywo
              └── @submit → onSurveySubmit (TODO: zapis danych) ← dane badania nie są jeszcze nigdzie utrwalane
```

**Punkt styku (docelowy):** poziom drżenia z toru A ma automatycznie przełączać
`uiMode` w `MedicalSurvey` (dziś robi to ręcznie badacz). Prop `initial-mode`
i zdarzenie `mode-change` są już przygotowane pod to połączenie.

---

## 5. Dane — co faktycznie pobrano

Zweryfikowany stan `ml-pipeline/data/`:

- **Pobrane sygnały:** 336 plików `.txt` w `data/raw/pads_dataset/movement/timeseries/`.
- **Pobrane etykiety:** **tylko 5** plików `patient_*.json` (001–005). To wąskie gardło —
  `convert` tworzy CSV wyłącznie dla pacjentów, którzy mają plik `patient_*.json`.
- **Skonwertowane:** 30 plików CSV (`data/raw/pads_csv/`) = 5 pacjentów × 3 zadania
  (Relaxed, RelaxedTask, StretchHold) × 2 nadgarstki.
- **`labels.csv`** (30 wierszy), rozkład klas:

  | label | condition | liczba plików | pacjenci |
  |---|---|---|---|
  | 0 | Healthy | 12 | 2 |
  | 1 | Parkinson's | 12 | 2 |
  | 2 | Other Movement Disorders | 6 | 1 |

- **`windows.npz`** (po preprocessingu): `X = (290, 300, 6)`, `fs = 100 Hz`,
  kanały `[accel_x..z, gyro_x..z]`, 5 pacjentów. Rozkład okien: klasa 0 → 116,
  klasa 1 → 116, klasa 2 → 58.

> **Wniosek:** potok zadziałał na realnych danych PADS i wszystkie 3 klasy są
> obecne, ale **5 pacjentów to zbiór czysto demonstracyjny**. Split per-pacjent
> przy 5 osobach daje ~1 pacjenta na walidację — wyniki nie są miarodajne.

---

## 6. Model — co wytrenowano

- Architektura: LSTM(64, return_sequences) → Dropout → LSTM(32) → Dropout →
  Dense(32, relu) → Dense(3, softmax).
- Wejście: `(300, 6)` = 3 s przy 100 Hz, 6 kanałów.
- Wyjście: **3 klasy diagnozy** (0=zdrowy, 1=Parkinson, 2=inne), **nie** poziom nasilenia.
- Wytrenowany na 290 oknach od 5 pacjentów → model o charakterze smoke-testu.

---

## 7. Czego brakuje — lista zadań (priorytetowo)

### Krytyczne (blokują sensowny wynik badawczy)
1. **Dociągnąć więcej danych PADS** — przede wszystkim brakujące `patient_*.json`
   dla pobranych już 336 sygnałów (obecnie tylko 5 ma etykiety), docelowo kolejni
   pacjenci. Użyć `download_pads.py fetch --subjects "..."`.
2. **Przetrenować model** na zbiorze wystarczającej wielkości i zweryfikować
   metryki per-klasa (balanced accuracy, recall klasy mniejszościowej).
3. **Eksport do TensorFlow.js** (osobny venv `requirements-tfjs.txt`) i skopiowanie
   `model.json` + `*.bin` + `normalization.json` do `mobile-app/public/model/`.
4. **Spiąć tor A z torem B** — poziom drżenia z modelu ma automatycznie przełączać
   `uiMode` w `MedicalSurvey` (dziś ręcznie w panelu badacza).

### Ważne (spójność i poprawność)
5. **Ujednolicić `fs` i rozmiar okna** offline ↔ mobile:
   - offline: `fs=100 Hz`, okno 300 próbek (wejście modelu `(300, 6)`);
   - mobile: `preprocess`/`signal.js` domyślnie `fs=50 Hz`, okno 150.
   - Aplikacja musi dostarczać okna `(300, 6)` zgodne z wejściem modelu.
6. **Rozstrzygnąć rozbieżność: model diagnozy (3 klasy) vs adaptacja wg nasilenia**
   (`useAdaptiveUI` oczekuje 4 poziomów 0–3). Model nie przewiduje nasilenia —
   docelowo wymaga etykiet w skali MDS-UPDRS (osobny etap wg README).
7. **Jednostki żyroskopu** — web `devicemotion` daje deg/s; upewnić się, że są
   spójne z jednostkami w danych treningowych PADS (uwaga odnotowana w `useMotionSensors.js`).
8. **Utrwalanie wyników badania HCI** — `onSurveySubmit` w `App.vue` to dziś
   `console.log`. Potrzebny zapis lokalny / eksport CSV / wysyłka do backendu.

### Porządkowe
9. **Zacommitować nietrackowane pliki** `MedicalSurvey.vue` i `useUxMetrics.js`
   (to bieżąca, niezapisana w historii praca).
10. **Odświeżyć sekcję „Stan projektu" w `README.md`** (datowana 2026-10-03,
    nie opisuje pivotu w stronę badania HCI ani faktycznego treningu na 5 pacjentach).
11. **Zdecydować los osieroconych komponentów** (`AdaptiveButton.vue`,
    `SurveyForm.vue`, `TremorStatus.vue`) — podpiąć pod tor A albo usunąć.
12. **`.ailoc2-metrics/`** — nietrackowany katalog narzędziowy; rozważyć dopisanie do `.gitignore`.

---

## 8. Znane niespójności i ryzyka

| # | Ryzyko | Skąd wynika |
|---|---|---|
| R1 | Model = diagnoza (3 klasy), a UI adaptuje się wg nasilenia (4 poziomy) | `train.py --num-classes 3` vs `SEVERITY_PROFILES` (4 wpisy) w `useAdaptiveUI.js` |
| R2 | Niespójne `fs`/okno offline (100 Hz, 300) vs mobile (50 Hz, 150) | `PADS_FS=100` w pipeline vs `PreprocessConfig.fs=50` i domyślne w `signal.js` |
| R3 | Filtr offline zero-phase (`filtfilt`) vs mobilny causalny (biquad RBJ) — różne charakterystyki fazowe | `preprocess.py` vs `signal.js` (świadoma decyzja, ale wymaga walidacji parzystości pasma) |
| R4 | Model statystycznie nieistotny (5 pacjentów) | tylko 5 `patient_*.json` pobranych |
| R5 | Dane badania HCI nie są utrwalane | `onSurveySubmit` = `console.log` |

---

## 9. Jak uruchomić (skrót)

### ml-pipeline
```bash
cd ml-pipeline
py -3.11 -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

python download_pads.py fetch   --root data/raw/pads_dataset        # pobranie (hotspot, bez VPN)
python download_pads.py inspect --root data/raw/pads_dataset        # weryfikacja formatu
python download_pads.py convert --root data/raw/pads_dataset --out data/raw/pads_csv
python preprocess.py --input data/raw/pads_csv --output data/processed --fs 100 --labels data/raw/pads_csv/labels.csv
python train.py --data data/processed/windows.npz --out models --num-classes 3

# eksport TF.js — osobny venv (konflikt uvloop na Windows):
py -3.11 -m venv .venv-tfjs && source .venv-tfjs/Scripts/activate
pip install -r requirements-tfjs.txt
tensorflowjs_converter --input_format=keras models/tremor_lstm.keras models/tfjs_model
# następnie skopiuj models/tfjs_model/* + models/normalization.json → mobile-app/public/model/
```

### mobile-app
```bash
cd mobile-app
npm install
npm run dev            # podgląd w przeglądarce (Network URL do testu na telefonie)
npm run build          # build produkcyjny
npm run cap:sync && npm run cap:android   # Android (wymaga Android Studio)
```

> Ograniczenia środowiska firmowego (VPN/allowlista), tryb demo, podgląd na
> telefonie i cytowanie zbioru PADS — patrz `README.md`.
