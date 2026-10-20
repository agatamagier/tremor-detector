# Jak uruchomić trening — instrukcja dla początkujących

> Ten plik tłumaczy **krok po kroku, bardzo prostym językiem**, jak ręcznie
> przejść całą drogę od surowych danych do wytrenowanego modelu i oceny jego
> jakości. Nie zakłada wiedzy o uczeniu maszynowym. Jeśli coś jest oczywiste —
> pomiń. Jeśli coś nie działa — patrz sekcja **9. Gdy coś nie działa**.

---

## 0. O co w ogóle chodzi (w jednym akapicie)

Chcemy, żeby program na podstawie **drgań ręki** (zmierzonych czujnikiem ruchu —
akcelerometrem i żyroskopem) rozpoznał, czy ktoś ma **drżenie parkinsonowskie**.
Parkinson daje charakterystyczne drżenie spoczynkowe o częstotliwości **4–6 Hz**
(czyli ~4–6 „drgnięć" na sekundę). Cała „magia" polega na tym, żeby z nagrania
ruchu policzyć, **ile energii drgania jest w paśmie 4–6 Hz**, i na tej podstawie
nauczyć prosty model odróżniać chorych od zdrowych.

Droga wygląda tak:

```
surowe nagrania (.txt)  →  BUILD  →  paczka okien (.npz)  →  TRENING  →  model  →  OCENA
      (czujnik)            (cięcie          (gotowe do          (uczenie)   (plik)   (czy dobry?)
                            na kawałki)       uczenia)
```

---

## 1. Zanim zaczniesz: terminal i środowisko

**Gdzie jesteś.** Wszystkie komendy uruchamiasz w folderze `ml-pipeline`.
W terminalu (Git Bash) wejdź do niego:

```bash
cd "C:/Users/AMalachowska/Desktop/New folder/TREMOR_DETECTOR/ml-pipeline"
```

**Środowisko (venv).** „venv" to taki **osobny plecak z narzędziami** (Python +
biblioteki), żeby nie mieszać z resztą komputera. Mamy go już gotowego w folderze
`.venv`. Nie musisz nic instalować. Masz dwie opcje:

- **Opcja A (najprościej):** przed każdą komendą pisz pełną ścieżkę do Pythona
  z plecaka:
  ```bash
  .venv/Scripts/python.exe nazwa_skryptu.py ...
  ```
- **Opcja B (wygodniej):** raz „włącz" plecak, potem pisz samo `python`:
  ```bash
  source .venv/Scripts/activate
  python nazwa_skryptu.py ...
  ```

> W tej instrukcji używam **Opcji A** (pełna ścieżka), bo zawsze działa.

---

## 2. Skąd się biorą dane (pobieranie)

Dane to zbiór **PADS** z portalu PhysioNet (nagrania ruchu nadgarstków pacjentów).
Każdy pacjent ma kilka plików `.txt` — po jednym na **zadanie** (np. „Relaxed" =
ręka rozluźniona) i **nadgarstek** (lewy/prawy). Nazwa pliku mówi wszystko:

```
005_RelaxedTask_LeftWrist.txt
 │        │          └── nadgarstek
 │        └── zadanie (co pacjent robił)
 └── numer pacjenta
```

Nie pobieramy całości (to 770 MB!). Pobieramy **tylko to, czego potrzebujemy**:
pacjentów z naszej listy i **tylko zadania spoczynkowe** (Relaxed, RelaxedTask) —
bo tam drżenie Parkinsona jest najczystsze.

Standardowe pobieranie wybranych pacjentów/zadań:

```bash
.venv/Scripts/python.exe download_pads.py fetch \
    --root data/raw/pads_dataset \
    --tasks Relaxed \
    --subjects "1-50"
```

- `--tasks Relaxed` → weź zadania, których nazwa zawiera „Relaxed" (łapie też
  „RelaxedTask"). Pominie StretchHold i zadania kinetyczne.
- `--subjects "1-50"` → pacjenci od 1 do 50. Można też `"1,2,3"` albo `"1-10,20"`.

> **To pobieranie jest „wznawialne":** jak się zerwie internet, uruchom ponownie
> tę samą komendę — pominie to, co już pobrał, i dociągnie resztę. Nie zacznie od zera.

Sprawdzenie, co pobrałaś (zanim pójdziesz dalej — **obowiązkowe**):

```bash
.venv/Scripts/python.exe download_pads.py inspect --root data/raw/pads_dataset
```

To wypisze: ilu masz pacjentów, jakie zadania, ile kolumn ma sygnał (ma być 7:
czas + 3 osie akcelerometru + 3 osie żyroskopu) — czyli czy dane wyglądają zdrowo.

---

## 3. BUILD — czyli „pokrój dane na kawałki gotowe do uczenia"

### Co to jest „build" i po co?

Model nie potrafi uczyć się z całego, długiego nagrania naraz. Musimy je **pociąć
na krótkie kawałki** (tzw. **okna**) i każdy kawałek **przefiltrować**, żeby
zostało tylko drżenie (3–12 Hz), bez „powolnego kołysania ręką" i bez szumu.
To właśnie robi „build":

1. **Resampling** — wyrównuje nagranie do równego tempa 100 próbek/s (czujnik
   nie próbkuje idealnie równo).
2. **Filtr pasmowy 3–12 Hz** — wycina wszystko poza pasmem drżenia (jak equalizer
   w muzyce: zostawiamy tylko „środek").
3. **Okienkowanie** — tnie nagranie na **okna po 3 sekundy**, zachodzące na siebie
   w 50% (czyli co 1,5 s nowe okno). Z jednego ~20-sekundowego nagrania powstaje
   kilkanaście okienek — więcej przykładów do nauki.

Wynik to jeden plik `.npz` (spakowana „paczka" z tablicami liczb): okna + etykiety
(0 = zdrowy, 1 = chory) + numer pacjenta dla każdego okna.

### Dwa sposoby na build

**Sposób ogólny** (wszyscy pacjenci, klasy 0/1/2 wg diagnozy) — dwa kroki:

```bash
# 1) zamień surowe .txt na nasz format CSV + plik z etykietami
.venv/Scripts/python.exe download_pads.py convert \
    --root data/raw/pads_dataset \
    --out data/raw/pads_csv \
    --tasks Relaxed

# 2) pokrój CSV na okna
.venv/Scripts/python.exe preprocess.py \
    --input data/raw/pads_csv \
    --output data/processed \
    --fs 100 \
    --labels data/raw/pads_csv/labels.csv
```

**Sposób docelowy dla NASZEGO eksperymentu** (tylko PD „tremor-dominant" vs zdrowi,
tylko spoczynek) — jeden krok, skrypt `build_td_rest.py`:

```bash
.venv/Scripts/python.exe build_td_rest.py \
    --cohort cohort_td_healthy.json \
    --root data/raw/pads_dataset \
    --out data/processed/windows_td_rest.npz
```

- `--cohort cohort_td_healthy.json` → lista pacjentów: kto jest „TD-PD" (drżący
  Parkinson), a kto zdrowy. Tę listę zbudowaliśmy wcześniej, czytając opis choroby
  każdego pacjenta.
- Skrypt sam wybierze **tylko tych pacjentów** i **tylko zadania spoczynkowe**,
  ponadaje etykiety (zdrowy → 0, TD-PD → 1) i zapisze gotową paczkę okien.

Na końcu wypisze m.in.: ile okien, ilu pacjentów, rozkład klas i — ważne —
czy ktoś z listy **nie ma jeszcze pobranych** danych (do dociągnięcia).

---

## 4. TRENING — czyli „naucz model odróżniać chorych od zdrowych"

Komenda:

```bash
.venv/Scripts/python.exe train_freq.py \
    --data data/processed/windows_td_rest.npz \
    --out models_freq_td_rest \
    --num-classes 2
```

Co oznaczają opcje (najważniejsze):

| Opcja | Co robi | Domyślnie |
|---|---|---|
| `--data` | która paczka okien (.npz) | `windows.npz` |
| `--out` | gdzie zapisać gotowy model | `models_freq` |
| `--num-classes` | ile klas: **2** = chory/zdrowy, 3 = zdrowy/PD/inne | 2 |
| `--epochs` | ile razy model przejrzy dane (więcej = dłużej) | 80 |
| `--val-split` | jaka część pacjentów idzie na „egzamin" (0.2 = 20%) | 0.2 |
| `--seed` | ziarno losowości (ten sam = powtarzalny wynik) | 42 |

### Co ten trening robi „w środku" (prosto)

1. Dla każdego 3-sekundowego okna liczy **cechy częstotliwościowe** — przede
   wszystkim: **jaki udział energii jest w paśmie 4–6 Hz** (parkinsonowskim).
   To kluczowe: zamiast dawać modelowi surowy „drut" sygnału, dajemy mu gotową
   podpowiedź „ile tu jest drżenia 4–6 Hz".
2. **Dzieli pacjentów** na uczących się i egzaminowanych — tak, żeby **ten sam
   pacjent nie był w obu grupach** (inaczej model „ściągałby" i wyniki byłyby
   zawyżone — patrz słowniczek: *wyciek danych*).
3. **Uczy mały model** (sieć zwana MLP) rozpoznawać wzorzec.
4. **Egzaminuje** go na nieznanych pacjentach i wypisuje oceny (sekcja 6).

### Co zobaczysz na ekranie

- Najpierw test „na sucho": **udział 4–6 Hz** osobno dla zdrowych i chorych.
  Jeśli u chorych jest wyraźnie wyższy — cecha ma sens (dobry znak!).
- Potem linijki `Epoch 1/80 ...` — model się uczy; `accuracy` powinno rosnąć.
- Na końcu **macierz pomyłek** i **metryki** (jak je czytać — sekcja 6).

---

## 5. OCENA „na pacjenta" — czyli diagnoza całego człowieka, nie pojedynczego okna

Lekarz diagnozuje **pacjenta**, a nie pojedyncze 3 sekundy nagrania. Dlatego
uśredniamy decyzje modelu ze **wszystkich okien danego pacjenta** w jedną
diagnozę:

```bash
.venv/Scripts/python.exe eval_patient_level.py \
    --data data/processed/windows_td_rest.npz \
    --model models_freq_td_rest \
    --num-classes 2
```

Wypisze dla każdego pacjenta: prawdę, predykcję i „pewność", oraz łączną trafność
na poziomie pacjenta.

---

## 6. Jak czytać wyniki (dla początkujących)

**Macierz pomyłek** — tabelka „co było naprawdę" (wiersze) vs „co model zgadł"
(kolumny):

```
        pred0   pred1
 true0   30      5      ← 30 zdrowych rozpoznanych dobrze, 5 błędnie jako chorzy
 true1    4     28      ← 28 chorych dobrze, 4 błędnie jako zdrowi
```

- **Recall (czułość) danej klasy** = ile z faktycznych przypadków tej klasy
  złapaliśmy. Dla chorych: 28 / (28+4) = **0,875**. Czyli „na 100 chorych wykryliśmy 87".
- **Precision (precyzja)** = gdy model mówi „chory", jak często ma rację.
- **Balanced accuracy** = **średnia z czułości obu klas**. Używamy jej, bo jest
  **uczciwa przy nierównych grupach**. Przykład: gdyby 90% ludzi było zdrowych,
  model mówiący ZAWSZE „zdrowy" miałby 90% „zwykłej" trafności — a jest
  bezużyteczny. Balanced accuracy to zdemaskuje (wyjdzie ~0,5 = jak rzut monetą).

**Punkty odniesienia:**
- `0,5` = rzut monetą (model nic nie umie).
- `~0,89` (nasz eksperyment 7) = bardzo dobrze, ale na małej próbce (patrz k-fold).
- `1,0` = idealnie (w praktyce podejrzane — często znak błędu/wycieku danych).

---

## 7. K-FOLD — czyli „sprawdź model uczciwie, nie na loterii"

### Problem, który rozwiązuje

Gdy mamy mało pacjentów, wynik **zależy od tego, kto akurat trafił na egzamin**.
Jak wylosujemy 5 „łatwych" pacjentów — wyjdzie super. Jak 5 „trudnych" — słabo.
To **loteria**, a nie rzetelna ocena.

### Pomysł k-fold (walidacja krzyżowa) — analogia

Wyobraź sobie klasę 25 uczniów i chcesz sprawdzić, jak dobrze **nauczyłaś**,
ale masz tylko tych 25 osób. Robisz tak:

1. Dzielisz klasę na **5 grup po 5 osób**.
2. Uczysz na 4 grupach (20 osób), **egzaminujesz 5. grupę**.
3. Powtarzasz **5 razy** — za każdym razem inna grupa jest egzaminowana.
4. **Każdy uczeń był dokładnie raz egzaminowany** (na materiale, którego przy jego
   egzaminie model nie widział). Bierzesz **średnią z 5 egzaminów**.

To właśnie **5-fold cross-validation**. „Fold" = jedna z tych grup. Dzięki temu:
- każdy pacjent jest oceniony raz, więc używamy **wszystkich** danych do oceny,
- dostajesz **średnią ± rozrzut** — widać, czy wynik jest stabilny, czy to loteria.

> U nas dzielimy **po pacjentach** (nie po oknach!), żeby okna jednego pacjenta
> nie trafiły naraz do uczenia i egzaminu (to byłby wyciek danych).

### Komenda

```bash
.venv/Scripts/python.exe kfold_freq.py \
    --data data/processed/windows_td_rest.npz \
    --num-classes 2 \
    --folds 5
```

Zobaczysz wynik każdego foldu, a na końcu:

```
Balanced accuracy (okno): 0.84 ± 0.05   (foldy: [0.81, 0.88, 0.79, 0.86, 0.86])
```

- liczba po `±` to **rozrzut**: mały = wynik stabilny (dobrze!), duży = loteria.
- plus zbiorcza macierz pomyłek na poziomie pacjenta (każdy pacjent raz).

**To jest wynik, który wpisuje się do pracy / artykułu** — nie pojedynczy split.

---

## 8. Cała droga w skrócie (ściąga do skopiowania)

```bash
cd "C:/Users/AMalachowska/Desktop/New folder/TREMOR_DETECTOR/ml-pipeline"

# (raz) pobierz dane spoczynkowe wybranych pacjentów
.venv/Scripts/python.exe download_pads.py fetch --root data/raw/pads_dataset --tasks Relaxed --subjects "1-469"

# 1. BUILD — pokrój na okna (nasz eksperyment: TD-PD vs zdrowi, spoczynek)
.venv/Scripts/python.exe build_td_rest.py --cohort cohort_td_healthy.json --root data/raw/pads_dataset --out data/processed/windows_td_rest.npz

# 2. TRENING
.venv/Scripts/python.exe train_freq.py --data data/processed/windows_td_rest.npz --out models_freq_td_rest --num-classes 2

# 3. OCENA na pacjenta
.venv/Scripts/python.exe eval_patient_level.py --data data/processed/windows_td_rest.npz --model models_freq_td_rest --num-classes 2

# 4. K-FOLD — rzetelna ocena
.venv/Scripts/python.exe kfold_freq.py --data data/processed/windows_td_rest.npz --num-classes 2 --folds 5
```

---

## 9. Gdy coś nie działa (najczęstsze problemy)

| Objaw | Co to znaczy / co zrobić |
|---|---|
| `The handshake operation timed out` przy pobieraniu | Zryw internetu. **Uruchom pobieranie jeszcze raz** — jest wznawialne, dociągnie resztę. |
| Pobieranie zabite „low on memory" | Za dużo rzeczy naraz. Zamknij inne programy i wznów pobieranie (pominie już pobrane). |
| `convert` tworzy puste wyjście | Brak plików `patient_*.json` (etykiet). Dociągnij etykiety pacjentów. |
| `Sygnał zbyt krótki ...` przy build | Nagranie krótsze niż jedno okno — skrypt je pomija, to normalne. |
| Log treningu to „krzaki"/`Binary file matches` | TensorFlow pisze znaki binarne. Zapisz log do pliku i czytaj przez `grep -a`, np. `komenda > log.txt 2>&1`, potem `grep -a accuracy log.txt`. |
| `ModuleNotFoundError` | Nie używasz Pythona z plecaka. Pisz `.venv/Scripts/python.exe ...` (Opcja A z sekcji 1). |
| Wynik = dokładnie `0,5` balanced accuracy | Model „się zapadł" (zgaduje jedną klasę). Patrz DZIENNIK_BADAWCZY — pomaga reprezentacja częstotliwościowa i czysty dobór grup. |

---

## 10. Słowniczek (najkrócej jak się da)

- **Okno** — kawałek nagrania o stałej długości (u nas 3 s), na którym model pracuje.
- **Hz (herc)** — liczba drgań na sekundę. Parkinson: ~4–6 Hz.
- **Cecha** — jedna liczba opisująca okno (np. „udział energii w 4–6 Hz").
- **Model / MLP** — program, który z cech uczy się przewidywać klasę; MLP to
  prosta sieć neuronowa.
- **Etykieta / klasa** — prawidłowa odpowiedź (0 = zdrowy, 1 = chory).
- **Epoka (epoch)** — jedno pełne przejście modelu przez dane uczące.
- **Trening vs walidacja** — dane do **uczenia** vs osobne dane do **egzaminu**.
- **Balanced accuracy** — średnia czułość po klasach; uczciwa przy nierównych grupach.
- **Recall (czułość)** — ile faktycznych chorych wykryliśmy.
- **Macierz pomyłek** — tabelka „prawda vs zgadnięte".
- **Wyciek danych (leakage)** — gdy model „podgląda" egzamin podczas nauki
  (np. okna tego samego pacjenta w train i val) → zawyżony, nieprawdziwy wynik.
- **K-fold (walidacja krzyżowa)** — wielokrotny egzamin, w którym każdy pacjent
  jest raz egzaminowany; daje średnią ± rozrzut zamiast loterii.
- **TD-PD** — „tremor-dominant" Parkinson: podtyp, w którym dominuje drżenie
  (w odróżnieniu od podtypów sztywnych, które prawie nie drżą).
```
