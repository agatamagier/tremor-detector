# Tremor Detector — Adaptive Mobile Interface for Parkinson's Hand Tremor

System **Edge AI**, który na żywo analizuje drżenie rąk i adaptuje interfejs
aplikacji mobilnej do jego nasilenia. Projekt badawczy:
*"Adaptive Mobile Interface Responding to Hand Tremor Severity in Patients with
Parkinson's Disease"*.

Monorepo złożone z dwóch części:

| Część | Technologie | Rola |
|---|---|---|
| **`ml-pipeline/`** | Python 3.11, TensorFlow/Keras, SciPy, pandas | Pobranie danych PADS, filtr Butterwortha (3–12 Hz), okienkowanie, trening LSTM, eksport do TensorFlow.js |
| **`mobile-app/`** | Vue 3, Vite, Tailwind, Capacitor, TensorFlow.js | Aplikacja mobilna: ankieta, inferencja modelu na urządzeniu, adaptacja UI (48→72 dp), filtrowanie jitteru dotyku |

---

## Stan projektu (aktualizacja: 2026-10-03)

**Zrobione:**
- ✅ Struktura monorepo.
- ✅ `ml-pipeline`: `preprocess.py` (filtr + okienkowanie + etykiety), `train.py`
  (LSTM, 3 klasy, split per-pacjent, eksport), `download_pads.py` (pobieranie +
  adapter PADS → nasz CSV).
- ✅ `mobile-app`: pełny scaffold Vue 3 + Capacitor, potok czujniki→filtr→okno→
  model→adaptacja UI, tryb demo. Buduje się (`npm run build`).
- ✅ Środowisko Python 3.11 + rdzeń bibliotek (TensorFlow 2.16).

**Do zrobienia (następne kroki):**
- ⏳ Pobrać zbiór PADS z PhysioNet (wymaga sieci bez firmowego VPN — patrz niżej).
- ⏳ Uruchomić `download_pads.py inspect` i **zweryfikować realny format** danych,
  w razie potrzeby dostroić adapter (nazwy kanałów / kolumn `.txt`).
- ⏳ Konwersja → preprocessing → trening → eksport modelu do `mobile-app/public/model/`.
- ⏳ Balansowanie klas w `train.py` (PADS jest niezbalansowany: dużo PD, mało „inne”).
- ⏳ Docelowo: zestawienie nasilenia drżenia ze skalą **MDS-UPDRS** (osobny etap).

---

## Architektura / przepływ danych

```
                        ┌─────────────────── ml-pipeline (offline, Python) ───────────────────┐
 PhysioNet PADS  ──►  download_pads.py  ──►  preprocess.py  ──►  train.py  ──►  tfjs_model/
 (accel+gyro,          (convert do CSV      (Butterworth        (LSTM 3 klasy,   model.json
  ~100 Hz)              + labels.csv)        3–12 Hz, okna        split/pacjent)  + wagi .bin
                                             3 s / 50% overlap)                   + normalization.json
                                                                                       │
                        ┌──────────────── mobile-app (on-device, Vue+TF.js) ───────────┼────────┐
 czujniki IMU  ──►  StreamingBandpass  ──►  WindowBuffer  ──►  useTremorModel  ◄────────┘
 (Capacitor/Web)    (3–12 Hz, causalny)    (okno 3 s)         (predict → klasa)
                                                                   │
                                                      useAdaptiveUI (histereza)
                                                                   │
                                      zmienne CSS: --touch-target 48→72px, odstępy, czcionka
                                      + touchFilter: EMA / dead-zone / debounce tapów
```

---

## Struktura katalogów

```
TREMOR_DETECTOR/
├── README.md                   # ten plik
├── ml-pipeline/
│   ├── requirements.txt        # rdzeń (numpy, pandas, scipy, tensorflow)
│   ├── requirements-tfjs.txt   # tensorflowjs — OSOBNY venv (konflikt uvloop na Win)
│   ├── download_pads.py        # download | inspect | convert (PADS → CSV + labels)
│   ├── preprocess.py           # filtr Butterwortha + okienkowanie + etykiety
│   ├── train.py                # LSTM, 3 klasy, split per-pacjent, eksport TF.js
│   └── data/
│       ├── raw/                # surowe CSV / pobrany PADS (ignorowane w git)
│       ├── processed/          # windows.npz (ignorowane)
│       └── models/ (models/)   # artefakty modelu (ignorowane)
└── mobile-app/
    ├── package.json            # Vue 3 + Vite + Tailwind + Capacitor + tfjs
    ├── capacitor.config.json
    ├── public/model/           # ← tu kopiujesz model.json + *.bin + normalization.json
    └── src/
        ├── App.vue             # spina cały potok
        ├── composables/        # useTremorModel, useMotionSensors, useAdaptiveUI
        ├── utils/              # signal.js (filtr+okna), touchFilter.js
        └── components/         # AdaptiveButton, SurveyForm, TremorStatus
```

---

## ml-pipeline — konfiguracja i uruchomienie

### Wymagania środowiska
- **Python 3.11** (TensorFlow NIE wspiera 3.13/3.14 — to był realny blocker w tym projekcie).
- Instalacja Pythona na Windows: `winget install Python.Python.3.11 --scope user`.

### Instalacja
```bash
cd ml-pipeline
py -3.11 -m venv .venv
source .venv/Scripts/activate       # Git Bash;  PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt      # rdzeń (BEZ tensorflowjs)
```

### Pełny workflow
```bash
# 1. Pobierz PADS (wymaga sieci bez firmowego VPN — np. hotspot telefonu)
python download_pads.py download --root data/raw/pads_dataset

# 2. ⚠️ OBOWIĄZKOWO zajrzyj w realną strukturę przed konwersją
python download_pads.py inspect  --root data/raw/pads_dataset

# 3. Konwersja PADS → nasz CSV + labels.csv
python download_pads.py convert  --root data/raw/pads_dataset --out data/raw/pads_csv

# 4. Filtr 3–12 Hz + okienkowanie (UWAGA: --fs 100, bo PADS próbkuje ~100 Hz!)
python preprocess.py --input data/raw/pads_csv --output data/processed \
    --fs 100 --labels data/raw/pads_csv/labels.csv

# 5. Trening LSTM (3 klasy) + próba eksportu
python train.py --data data/processed/windows.npz --out models --num-classes 3
```

### Kluczowe parametry sygnału (muszą być spójne offline ↔ mobile)
- Pasmo drżenia: **3–12 Hz** (bandpass Butterworth).
- Okno: **3 s**, overlap **50%**.
- `fs`: **100 Hz** dla PADS (w aplikacji mobilnej domyślnie 50 Hz — do ujednolicenia
  przy integracji realnego modelu).

### Etykiety (cel: detekcja PD)
Z metadanych pacjenta PADS (`condition`):
`Healthy → 0`, `Parkinson's → 1`, `Essential Tremor / MS / Atypical / Other → 2`.
Nasilenie drżenia (skala UPDRS) to osobny, późniejszy etap.

### Eksport do TensorFlow.js (osobne środowisko!)
`tensorflowjs` na Windows ciągnie `uvloop`, który się nie buduje — dlatego jest
w osobnym pliku i powinien iść do dedykowanego venv:
```bash
py -3.11 -m venv .venv-tfjs
source .venv-tfjs/Scripts/activate
pip install -r requirements-tfjs.txt
tensorflowjs_converter --input_format=keras models/tremor_lstm.keras models/tfjs_model
```
Jeśli i to padnie na Windows — alternatywą jest konwersja w WSL/Linux lub Google Colab.

---

## mobile-app — konfiguracja i uruchomienie

### Instalacja i dev
```bash
cd mobile-app
npm install
npm run dev            # podgląd w przeglądarce; Network URL do testu na telefonie
```

### Podgląd na telefonie
- **Ten sam Wi-Fi**: otwórz `http://<IP-laptopa>:5173/` w przeglądarce telefonu.
- **Uwaga o czujnikach**: akcelerometr/żyroskop wymagają „secure context”
  (`localhost`, HTTPS lub apka natywna). Po zwykłym `http://LAN` działa tylko
  **tryb demo** (suwak nasilenia) — wystarcza do podglądu UI.

### Build natywny (Android, Capacitor)
Wymaga Android Studio (zawiera SDK + wbudowany JDK 17).
```bash
npm run build
npx cap add android
npm run cap:sync
npm run cap:android     # otwiera Android Studio → Run ▶
```

### Podłączenie modelu
Skopiuj do `mobile-app/public/model/`: `model.json`, pliki `*.bin`,
`normalization.json`. Bez nich apka startuje w trybie demo.

### Jak działa adaptacja UI
- `useAdaptiveUI` mapuje klasę modelu na profil UI (rozmiar celu dotykowego
  48→72 px, odstępy, skala czcionki, siła filtrów dotyku) z **histerezą**
  (zmiana poziomu wymaga kilku zgodnych predykcji — brak migotania).
- `touchFilter.js`: wygładzanie EMA pozycji, strefa martwa, debounce „tapów”.

---

## ⚠️ Ograniczenia środowiska (ważne przy kontynuacji)

Projekt powstawał na **firmowym laptopie z VPN i allowlistą hostów**:
- `github.com`, `physionet.org`, `cloudflare.com`, `imigitlab.uni-muenster.de`
  są **blokowane** przez firmowy firewall/VPN (odpowiedź `HTTP 000`).
- `registry.npmjs.org` i `pypi.org` **działają** (są na allowliście) — stąd
  `npm install` i `pip install` przechodzą.
- **Bez VPN laptop nie ma internetu** (VPN jest jedyną drogą na zewnątrz).
- Pobieranie danych (PADS/GitHub) i tunele HTTPS → rób przez **hotspot telefonu**
  lub sieć domową.
- Debugowanie USB telefonu bywa zablokowane politykowo — do podglądu UI użyj
  Wi-Fi/hotspotu.

---

## Źródło danych i cytowanie

Zbiór **PADS (Parkinson's Disease Smartwatch)**, PhysioNet v1.0.0:
<https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/>

Kod ewaluacyjny PADS (referencja formatu danych):
<https://imigitlab.uni-muenster.de/published/pads-project>

```
@article{varghese2024machine,
  title={Machine Learning in the Parkinson's disease smartwatch (PADS) dataset},
  author={Varghese, Julian and Brenner, Alexander and Fujarski, Michael and
          van Alen, Catharina Marie and Plagwitz, Lucas and Warnecke, Tobias},
  journal={npj Parkinson's Disease}, volume={10}, number={1}, pages={9}, year={2024}
}
```
