# Plan kontynuacji — Tremor Detector

> Tryb: **pełna sekwencja** (oba tory domknięte w kolejności zależności, bez skrótów).
> Stan wyjściowy i uzasadnienia — patrz [`DOKUMENTACJA.md`](./DOKUMENTACJA.md).
> Utworzono: 2026-10-19.

## Ścieżka krytyczna (jednym zdaniem)

**Dane (469 pacjentów) → preprocessing → realny model → eksport TF.js → integracja Edge AI z ankietą → utrwalanie wyników → protokół badania.**
Fazy 0 i 5 mogą biec równolegle do ścieżki krytycznej.

```
F0 higiena repo ─┐
                 ├─(równolegle)
F1 dane ─► F2 trening ─► F3 eksport TF.js ─► F4 integracja ─► F6 protokół + walidacja ─► F7 (opcja) UPDRS
                 └─ F5 utrwalanie wyników HCI ─(równolegle, wpina się w F4/F6)
```

---

## Faza 0 — Higiena repozytorium (≈30 min, równolegle, bez zależności)

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 0.1 | Zacommitować nietrackowane `mobile-app/src/components/MedicalSurvey.vue` i `mobile-app/src/composables/useUxMetrics.js` | `git status` czysty dla tych plików |
| 0.2 | Dodać `.ailoc2-metrics/` do `.gitignore` | katalog nie pojawia się w `git status` |
| 0.3 | Odświeżyć sekcję „Stan projektu" w `README.md` (data, pivot HCI, realny trening) | README zgodny z `DOKUMENTACJA.md` |

**Uwaga:** dane (`data/`), `models/`, `node_modules/`, `dist/`, `android/` są już ignorowane — nie commitować.

---

## Faza 1 — Dane PADS (KRYTYCZNA, w toku)

Cel: komplet zadań drżenia dla **wszystkich 469 pacjentów** + etykiety.

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 1.1 | **Pobranie** — `download_pads.py fetch --root data/raw/pads_dataset` (uruchomione w tle; ~2814 sygnałów + 469 `patient_*.json`, ~490 MB do dociągnięcia). Wznawialne. | 469 `patient_*.json` + ~2814 `.txt` w `data/raw/pads_dataset/` |
| 1.2 | **Weryfikacja formatu** — `download_pads.py inspect --root data/raw/pads_dataset` | potwierdzone: 7 kolumn `[Time, accel xyz, gyro xyz]`, rozkład etykiet pacjentów |
| 1.3 | **Konwersja** — `download_pads.py convert --root data/raw/pads_dataset --out data/raw/pads_csv` | `labels.csv` z ~469 pacjentami; log rozkładu klas 0/1/2 |

**Decyzja D1 (po 1.3):** realny rozkład klas PADS jest niezbalansowany (dużo PD).
Zanotować liczności 0/1/2 — wejście do strategii balansowania w F2.

**Ryzyka:** łącze PhysioNet bywa wolne; firmowy VPN potrafi blokować host
(teraz osiągalny). Przy przerwaniu — ponowić `fetch` (skip-if-exists).

---

## Faza 2 — Preprocessing + trening realnego modelu (KRYTYCZNA)

Zależy od: F1.

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 2.1 | **Preprocessing** — `preprocess.py --input data/raw/pads_csv --output data/processed --fs 100 --labels data/raw/pads_csv/labels.csv` | `windows.npz` z setkami/tysiącami okien, >100 pacjentów, 3 klasy obecne |
| 2.2 | **Trening** — `train.py --data data/processed/windows.npz --out models --num-classes 3` (balansowanie klas włączone) | model zapisany; **raport per-klasa** (recall klasy 2 „inne" > 0, balanced accuracy w logu) |
| 2.3 | **Ocena jakości** — przejrzeć macierz pomyłek i balanced accuracy; w razie słabej klasy mniejszościowej rozważyć augmentację / więcej epok / inną architekturę | świadoma decyzja o akceptacji modelu albo iteracji |

**Decyzja D2 — klasy modelu vs adaptacja UI (ryzyko R1):**
model przewiduje **diagnozę (3 klasy)**, a `useAdaptiveUI` oczekuje **nasilenia (0–3)**.
Do czasu etapu UPDRS (F7) przyjąć jawne mapowanie na potrzeby demo, np.:
`0 zdrowy → UI standard`, `1 PD / 2 inne → UI adapted`. Udokumentować wybór.

**Decyzja D3 — fs/okno (ryzyko R2):** model ma wejście `(300, 6)` (3 s @ 100 Hz).
Aplikacja musi dostarczać okna o tym samym kształcie → w F4 ustawić próbkowanie/
okno na 100 Hz / 300 próbek (albo przetrenować model pod 50 Hz / 150 — wybrać jedno).

---

## Faza 3 — Eksport do TensorFlow.js i podłączenie modelu (KRYTYCZNA)

Zależy od: F2.

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 3.1 | Osobny venv: `py -3.11 -m venv .venv-tfjs` + `pip install -r requirements-tfjs.txt` | środowisko gotowe (konflikt `uvloop` obejdzie osobny venv) |
| 3.2 | `tensorflowjs_converter --input_format=keras models/tremor_lstm.keras models/tfjs_model` (fallback: WSL/Colab, jeśli padnie na Windows) | `models/tfjs_model/model.json` + `*.bin` |
| 3.3 | Skopiować `models/tfjs_model/*` + `models/normalization.json` → `mobile-app/public/model/` | pliki obecne w `public/model/` |
| 3.4 | Test ładowania: `useTremorModel.load()` + `predict()` na zaślepce okna | brak błędu ładowania, `predict` zwraca `{severity, probabilities}` |

---

## Faza 4 — Integracja torów: Edge AI → adaptacja ankiety (KRYTYCZNA)

Zależy od: F3. Realizuje decyzje D2 i D3.

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 4.1 | Ujednolicić `fs`/okno wg D3 w `signal.js` (`StreamingBandpass`, `WindowBuffer`) i w `useMotionSensors` | okno podawane do modelu ma kształt `(300, 6)` |
| 4.2 | Spiąć potok: `useMotionSensors` → `StreamingBandpass` → `WindowBuffer` → `useTremorModel.predict` → mapowanie D2 → przełączanie `uiMode` w `MedicalSurvey` (prop `initial-mode` / zdarzenia już przygotowane) | w aplikacji tryb UI zmienia się automatycznie wg predykcji |
| 4.3 | Histereza: wpiąć `useAdaptiveUI.reportPrediction` tak, by UI nie „migotało" | brak przełączeń przy pojedynczej odstającej predykcji |
| 4.4 | Zweryfikować **jednostki żyroskopu** (web deg/s vs dane treningowe PADS) (ryzyko R3/R4) | jednostki spójne albo jawnie przeskalowane |
| 4.5 | Dodać przełącznik „auto / ręczny" w panelu badacza (do badania potrzebny jest też tryb ręczny) | badacz może wymusić tryb lub oddać sterowanie modelowi |

**Uwaga środowiskowa:** czujniki wymagają „secure context" (localhost/HTTPS/natywnie).
Test realnej automatyki tylko na telefonie przez HTTPS lub build Capacitora.

---

## Faza 5 — Utrwalanie wyników badania HCI (równolegle do F2–F4)

Zależy od: nic (tor B już działa). Rozwiązuje ryzyko R5.

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 5.1 | `onSurveySubmit` w `App.vue` → trwały zapis (minimum: akumulacja w `localStorage`) | każdy zapis ankiety ląduje w pamięci trwałej |
| 5.2 | Eksport zebranych sesji do CSV/JSON (przycisk w panelu badacza) | plik z kolumnami: `mode, tct, missclicks, odpowiedzi, timestamp` |
| 5.3 | (opcjonalnie) wysyłka do backendu badawczego, jeśli będzie | — |

---

## Faza 6 — Protokół badania i walidacja na urządzeniu (KRYTYCZNA dla wyniku badawczego)

Zależy od: F4 + F5.

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 6.1 | Zdefiniować protokół: standard vs adapted, randomizacja/kontrbalans kolejności, liczba prób na uczestnika | spisany scenariusz sesji |
| 6.2 | Build i test na realnym telefonie (`npm run build` → Capacitor Android) | aplikacja działa natywnie, czujniki aktywne |
| 6.3 | Pilotaż na kilku osobach (bez pacjentów) — sanity metryk TCT/missclicks | dane eksportują się, metryki sensowne |
| 6.4 | (jeśli dotyczy) zgody/etyka przy udziale pacjentów | formalności dopełnione |

---

## Faza 7 — (opcjonalnie) Nasilenie drżenia wg MDS-UPDRS

Zależy od: F2. Osobny, dłuższy etap — zamienia „diagnozę" na „nasilenie".

| # | Zadanie | Kryterium ukończenia |
|---|---|---|
| 7.1 | Pozyskać/zmapować etykiety nasilenia (MDS-UPDRS) do okien | zbiór z etykietą porządkową nasilenia |
| 7.2 | Przetrenować model jako regresję/klasyfikację porządkową nasilenia | model nasilenia 0–3 |
| 7.3 | Zdjąć mapowanie zastępcze D2 — `useAdaptiveUI` dostaje realne nasilenie | pełna zgodność modelu z 4 profilami UI |

---

## Rejestr decyzji (do rozstrzygnięcia w trakcie)

| ID | Decyzja | Faza | Domyślne podejście |
|---|---|---|---|
| D1 | Strategia balansowania wobec realnego rozkładu klas | F2 | class weights (jest) + ew. augmentacja klasy mniejszościowej |
| D2 | Mapowanie klasy modelu (diagnoza) → tryb UI | F2/F4 | 0→standard, 1/2→adapted (do czasu F7) |
| D3 | Ujednolicenie fs/okno (100 Hz/300 vs 50 Hz/150) | F2/F4 | aplikacja dostarcza 100 Hz / okno 300 pod wejście modelu |

---

## Status wykonania (aktualizowany)

- [~] **F1.1 Pobranie danych** — w toku w tle (zadanie `b609uu46o`), ~490 MB, wznawialne.
- [ ] pozostałe fazy — niezaczęte.
