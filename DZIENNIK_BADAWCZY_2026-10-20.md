# Dziennik badawczy — sesja 2026-10-20

> Zapis przebiegu prac nad modelem detekcji drżenia parkinsonowskiego:
> napotkane problemy, ich diagnoza, zastosowane rozwiązania oraz wyniki
> kolejnych eksperymentów. Dokument przygotowany pod sekcje **Metodyka**,
> **Ograniczenia** i **Wnioski** artykułu / pracy doktorskiej.
>
> Zbiór danych: **PADS** (Parkinson's Disease Smartwatch), PhysioNet v1.0.0.
> Podzbiór eksploracyjny (eksp. 1–7): **88 pacjentów, 523 sygnały**.
> Kohorta docelowa (eksp. 8–9): **131 pacjentów** (52 TD-PD + 79 zdrowych), celowane
> pobranie samych zadań spoczynkowych z pełnych 469 etykiet.

---

## 1. Cel sesji

Doprowadzić potok ML od surowych danych PADS do wytrenowanego, ocenionego
modelu 3-klasowego (0=zdrowy, 1=Parkinson, 2=inne zaburzenia ruchu) i rzetelnie
ocenić jego jakość. W trakcie pojawiły się istotne problemy techniczne
(pozyskanie danych) i merytoryczne (jakość modelu), opisane poniżej.

---

## 2. Część I — Pozyskanie danych (problemy i naprawy)

### P1. Powtarzające się zabicia procesu z braku pamięci (OOM)
- **Objaw:** proces pobierania w tle był dwukrotnie zabijany przez system
  („system is running low on memory").
- **Przyczyna:** równoległe obciążenie (jednoczesny build frontendu + Python +
  trening) pod presją pamięci.
- **Naprawa:** pobieranie zaprojektowane jako **wznawialne** (zapis atomowy
  `.part` + `os.replace`, pomijanie już istniejących plików) — po każdym zabiciu
  wznawiano bez utraty postępu.

### P2. Limit czasu polecenia (2 min)
- **Objaw:** polecenia z `sleep 120` kończyły się kodem 143 (timeout 120 s).
- **Naprawa:** rezygnacja z aktywnego czekania; sprawdzanie stanu plików
  zamiast usypiania procesu.

### P3. Błąd pobierania manifestu (`ContentTooShortError`)
- **Objaw:** `got only 933524 out of 1413776 bytes` przy pobieraniu
  `SHA256SUMS.txt` (manifest 1,4 MB).
- **Przyczyna:** niestabilne łącze + pobieranie manifestu **bez ponawiania**.
- **Naprawa:** **pominięcie manifestu** i pobranie etykiet bezpośrednio po
  znanym wzorcu URL (`patients/patient_XXX.json`, ID 3-cyfrowe) z ponawianiem
  (4 próby, backoff) i pomijaniem istniejących. Efekt: 83 nowe etykiety, 0 błędów.

### P4. Problem strukturalny — „etykiety pobierane na końcu"
- **Objaw:** mimo setek pobranych sygnałów `convert` tworzył puste wyjście.
- **Przyczyna:** kolejność pobierania umieszczała pliki `patient_*.json`
  (etykiety) **na końcu** kolejki; proces ginął (OOM) w fazie sygnałów, zanim
  dotarł do etykiet. Bez `patient_*.json` konwersja nie ma czym etykietować.
- **Naprawa:** pobranie etykiet **najpierw i niezależnie** (P3) — dla kompletu
  88 pacjentów, których sygnały już były na dysku.

### Decyzja D-DANE: ograniczenie zakresu
Na prośbę zdecydowano **nie** pobierać wszystkich ~3283 rekordów, lecz pracować
na **88 pacjentach** (523 sygnały) jako zbiorze roboczym — wystarczającym do
uruchomienia i oceny potoku, znacznie solidniejszym niż pierwotne 5 pacjentów.

### Stan danych po naprawach
- 523 pliki `.txt` (sygnały), 88 plików `patient_*.json` (etykiety), 0 błędów.
- Format potwierdzony: 7 kolumn `[Time, accel x/y/z, gyro x/y/z]`, 2048 próbek
  (~20,5 s @ 100 Hz); część nagrań krótsza (min ~10 s / 1024 próbki).

---

## 3. Część II — Uruchomienie potoku

| Krok | Wynik |
|---|---|
| `inspect` | 7 kolumn potwierdzone; rozkład etykiet pacjentów: **0=14, 1=48, 2=26** |
| `convert` | 523 CSV, 0 pominiętych; rozkład plików: 0=84, 1=288, 2=151 |
| `preprocess --fs 100` | **5048 okien**, kształt **(5048, 300, 6)**, 88 pacjentów; okna: 0→812, 1→2775, 2→1461 |
| `train --num-classes 3` | model zapisany, ale **patologiczny** (patrz niżej) |

Rozkład zadań w sygnałach: Relaxed 174, RelaxedTask 175, StretchHold 174.

---

## 4. Część III — Problem kluczowy: „zapadanie się" modelu

### P5. Model 3-klasowy degeneruje do jednej klasy
- **Objaw:** balanced accuracy **0.419** (próg losowy 0.333); recall klasy 1
  (Parkinson) = **0.091** — model prawie zawsze przewidywał klasę 0 (zdrowy,
  recall 0.98).
- **Przyczyna:** zadanie jest trudne, a przy włączonych wagach klas
  (`{0:2.334, 1:0.586, 2:1.156}`) sieć LSTM minimalizuje stratę, **zgadując
  najwyżej ważoną klasę** zamiast uczyć się cech. Early stopping (monitor
  `val_loss`) dodatkowo przywrócił wagi z wczesnej epoki o najsilniejszym kolapsie.
- **Interpretacja:** to nie usterka potoku — potok działa poprawnie. To sygnał,
  że **reprezentacja wejścia** (surowe okno) nie niesie dla tego modelu
  wystarczającej informacji różnicującej.

---

## 5. Część IV — Systematyczne eksperymenty

Każdy eksperyment = jedna hipoteza, jedna zmiana, pomiar tej samej metryki
(balanced accuracy + recall PD), split **per-pacjent** (bez wycieku okien).

| # | Podejście | Zadanie | Okno | Balanced acc | Recall PD | Kolaps? |
|---|---|---|---|---|---|---|
| 1 | LSTM, surowy sygnał | 3 klasy | 3 s | 0.419 | 0.091 | **tak** |
| 2 | LSTM, surowy sygnał | PD vs reszta | 3 s | 0.520 | 0.075 | **tak** |
| 3 | LSTM, surowy sygnał | PD vs reszta | 5 s | 0.528 | 0.121 | **tak** |
| 4 | **Cechy częstotliwości (MLP)** | PD vs reszta | 3 s | **0.589** | **0.502** | **nie** |
| 5 | Cechy częst., agregacja per-pacjent | PD vs reszta | 3 s | 0.569 | — | nie |
| 6 | Cechy częstotliwości (MLP) | Zdrowy vs PD | 3 s | 0.568 | — | nie |
| 7 | **Cechy częst. — TD-PD vs zdrowi, SPOCZYNEK** (smoke-test, 27 pac.) | TD-PD vs zdrowi | 3 s | **0.891** | **0.990** | nie |
| 8 | jw. — **PEŁNA kohorta 131 pac.**, pojedynczy split | TD-PD vs zdrowi | 3 s | 0.607 | 0.382 | nie |
| 9 | jw. — **pełna kohorta + walidacja 5-krotna (k-fold)** | TD-PD vs zdrowi | 3 s | **0.688 ± 0.062** | 0.568 ± 0.169 | nie |

### Hipoteza A — liczba klas (eksp. 2)
Scalenie do 2 klas (PD vs nie-PD) wyrównało rozkład (2273 vs 2775), ale model
nadal się zapadał (recall PD 0.075). **Wniosek: nie liczba klas była problemem.**

### Hipoteza B — dłuższe okno (eksp. 3)
Okno 5 s (sekwencja 500 kroków). Ograniczenie danych: nagrania PADS trwają tylko
~20 s, więc dłuższe okno = mniej fragmentów (10 s → ~4× mniej, najkrótsze nagrania
wypadają) i **trudniejsza** sekwencja dla LSTM. Wynik 0.528, nadal kolaps.
**Wniosek: długość okna nie była problemem.**

### Hipoteza C — reprezentacja częstotliwościowa (eksp. 4) — PRZEŁOM
Zamiast surowego sygnału: **widmo mocy (PSD, metoda Welcha)** każdego okna,
cechy per kanał (6 kanałów × 6 cech = 36): moc w paśmie 3–12 Hz, moc i udział
pasma **4–6 Hz** (parkinsonowskiego), udział 6–12 Hz, częstotliwość dominująca,
centroid widma. Klasyfikator: mały MLP.
- **Model przestał się zapadać**; recall PD **0.075 → 0.502**; balanced acc **0.589**.
- **Wniosek: problemem była reprezentacja** — LSTM nie potrafił sam „odkryć"
  częstotliwości drżenia z surowych próbek; jawne cechy widmowe to naprawiły.

### Hipoteza D — agregacja per-pacjent (eksp. 5)
Uśrednienie prawdopodobieństw modelu po wszystkich oknach pacjenta. **Bez
poprawy** (0.569). Diagnoza: `p(PD)` dla niemal każdego pacjenta ≈ 0.47–0.51 —
model wisi na granicy decyzji; agregacja niepewnych głosów daje niepewny wynik.
**Sygnał jest realny, ale słaby.**

### Hipoteza E — usunięcie mylącej klasy „inne" (eksp. 6)
Zdrowy vs Parkinson (bez „inne zaburzenia ruchu"). Nadal tylko 0.568.
**Wniosek: słaby wynik nie wynika wyłącznie z klasy „inne"** — sygnał jest słaby
także w najłatwiejszym wariancie (brak drżenia vs wyraźne drżenie).

### Hipoteza F — czyste ujęcie: tremor-dominant + spoczynek (eksp. 7) — PRZEŁOM
Połączenie dwóch filtrów o podstawach klinicznych:
- **tylko PD tremor-dominant** (13 pacjentów) — usunięcie chorych akinetic-rigid /
  hypokinetic-rigid, którzy niemal nie drżą i zaszumiali etykietę „Parkinson",
- **tylko zadania spoczynkowe** (Relaxed + RelaxedTask, bez StretchHold) — gdzie
  drżenie spoczynkowe PD (4–6 Hz) jest najsilniejsze.

Zbiór: 1296 okien (672 zdrowi / 624 TD-PD), 27 pacjentów, zbalansowany.
- Balanced accuracy (okno): **0.891**; recall TD-PD **0.990**; recall zdrowych 0.792.
- Poziom pacjenta: **5/5** poprawnie (3 zdrowych, 2 TD-PD).
- **Marker 4–6 Hz wreszcie rozdziela klasy:** TD-PD **0.401** vs zdrowi **0.271**.

**Wniosek:** sygnał był obecny przez cały czas, ale **ukryty przez heterogeniczne
etykiety i mieszanie zadań**. Czyste ujęcie (zgodne z pierwotnym celem badania —
pacjenci tremor-dominant) i pomiar w spoczynku to odsłaniają.

> **⚠️ Ograniczenie smoke-testu:** tylko **5 pacjentów w walidacji**. Trafność 5/5
> jest zachęcająca, ale niepewność ogromna. Najmocniejszym dowodem jest tu
> **niezależny od modelu** test rozdzielności 4–6 Hz (0.40 vs 0.27), a nie sama
> accuracy. Twardą wartość da dopiero pełna kohorta + walidacja k-krotna.

### Hipoteza F, weryfikacja na PEŁNEJ kohorcie (eksp. 8–9) — wynik docelowy

Zrealizowano plan z zastrzeżenia powyżej: pozyskano **pełną kohortę 131 pacjentów**
(52 TD-PD + 79 zdrowych; celowane pobranie samych sygnałów spoczynkowych, 0 błędów)
i zbudowano czysty zbiór **6288 okien** (3792 zdrowi / 2496 TD-PD) skryptem
`build_td_rest.py` (filtr podtypu + filtr zadania + relabel w jednym miejscu).

**Eksp. 8 — pojedynczy split per-pacjent (26 val):** balanced accuracy **0.607**,
recall TD-PD 0.382. Spadek z 0.891 → 0.607 **potwierdza, że wynik 5/5 był
„loterią" małej próby** — dokładnie jak ostrzegało zastrzeżenie smoke-testu.

**Eksp. 9 — walidacja 5-krotna grupowa po pacjencie (`kfold_freq.py`):**

| Metryka | Wynik |
|---|---|
| Balanced accuracy (okno) | **0.688 ± 0.062** (foldy: 0.616 / 0.666 / 0.685 / 0.671 / 0.802) |
| Recall TD-PD (okno) | 0.568 ± 0.169 |
| **Trafność na poziomie pacjenta** | **0.809 (106/131)** |
| **Balanced accuracy (pacjent)** | **0.773** |

Zbiorcza macierz pomyłek na poziomie pacjenta (każdy pacjent oceniony dokładnie raz):

```
         pred0  pred1
 true0    75     4     → zdrowi: recall 0.949
 true1    21    31     → TD-PD:  recall 0.596
```

**Wnioski docelowe:**
1. **Twarda, uczciwa wartość to ~0.77 balanced accuracy na poziomie pacjenta**
   (nie 0.89). Mieści się w realistycznym celu 0.75–0.85 i nadaje się do publikacji.
2. **Agregacja per-pacjent REALNIE pomaga** (0.773 vs 0.688 okno) — w przeciwieństwie
   do eksp. 5 (słaby sygnał, brak poprawy). Na większej, czystej kohorcie sygnał
   jest dość silny, by uśrednienie wielu okien stabilizowało diagnozę pacjenta.
3. **Asymetria klas:** zdrowych wykrywamy bardzo dobrze (recall 0.95), TD-PD słabiej
   (0.60) — ~40% drżących PD umyka. Zgodne z rozkładem markera 4–6 Hz (patrz niżej):
   rozdziela się (0.266 vs 0.354), ale rozkłady mocno się nakładają. Klinicznie
   sensowne — nawet w podtypie tremor-dominant drżenie spoczynkowe bywa łagodne/zmienne.
4. **Rozrzut między foldami** (recall PD 0.36–0.87) pokazuje, jak bardzo pojedynczy
   split potrafi zmylić — i dlaczego k-fold jest tu konieczny, a nie opcjonalny.

### Test rozdzielności pasma 4–6 Hz (bezpośredni, bez modelu)
Udział energii w paśmie 4–6 Hz (uśredniony po kanałach):

| Porównanie | nie-PD / zdrowy | PD |
|---|---|---|
| PD vs reszta (eksp. 4) | 0.306 | 0.293 |
| Zdrowy vs PD (eksp. 6) | 0.263 | 0.293 |
| **TD-PD vs zdrowi, spoczynek (eksp. 7, 27 pac.)** | **0.271** | **0.401** |
| **TD-PD vs zdrowi, spoczynek (eksp. 8–9, pełna kohorta 131 pac.)** | **0.266** | **0.354** |

**Kluczowa obserwacja:** klasyczny marker 4–6 Hz jest w tych danych **tylko
ledwo widoczny**. W porównaniu „PD vs reszta" PD ma nawet *niższy* udział 4–6 Hz
(bo grupa „reszta" zawiera drżenie samoistne i inne zaburzenia drżące w podobnym
paśmie). Dopiero „Zdrowy vs PD" pokazuje oczekiwany kierunek (0.293 > 0.263),
ale różnica jest mała, a rozkłady mocno się nakładają.

#### Interpretacja „paradoksu": dlaczego PD miał *niższy* udział 4–6 Hz

W eksperymencie 4 grupa „PD" (0.293) wyszła z udziałem 4–6 Hz **niższym** niż
„reszta" (0.306) — co wygląda absurdalnie, skoro to Parkinson ma dawać drżenie
4–6 Hz. To był **fałszywy wynik z „brudnego" porównania**, z trzech przyczyn:

1. **Grupa „PD" zawierała chorych, którzy prawie nie drżą** — podtypy
   akinetic-rigid / hypokinetic-rigid (dominuje sztywność, nie drżenie). Mają
   mało energii w 4–6 Hz i **zaniżali średnią całej grupy PD**.
2. **Grupa „reszta" nie była „bez drżenia"** — zawierała drżenie samoistne i inne
   zaburzenia, które **też drżą w podobnym paśmie**, więc i ona świeciła w 4–6 Hz.
3. **Zadania były mieszane** (w tym posturalny StretchHold), gdzie drżenie
   spoczynkowe PD słabnie.

**Analogia:** to jak sprawdzać, czy „sportowcy są wyżsi", wrzucając do grupy
sportowców **dżokejów** (niscy), a do nie-sportowców **kibiców koszykówki**
(wysocy). Różnica w średnim wzroście znika — mimo że koszykarze naprawdę są
wysocy. Tak samo marker 4–6 Hz „znikł", bo grupy były źle dobrane.

**Rozwiązanie (eksp. 7):** po oczyszczeniu — tylko tremor-dominant PD (faktycznie
drżący) vs zdrowi (niedrżący), tylko spoczynek — marker wrócił dokładnie jak w
podręczniku: **TD-PD 0.401 vs zdrowi 0.271**. Wniosek: marker 4–6 Hz jest
**prawdziwy**; wcześniej był **ukryty** przez heterogeniczne etykiety i mieszanie
zadań, a nie nieobecny.

---

## 6. Część V — Wnioski i ograniczenia (pod sekcję *Limitations*)

1. **Jakość zależy krytycznie od doboru grup i agregacji.** „Brudne" ujęcie
   (heterogeniczne etykiety, mieszane zadania, pojedynczy nadgarstek) daje sufit
   ~0.57–0.59 balanced accuracy. Po oczyszczeniu (TD-PD vs zdrowi, spoczynek, pełna
   kohorta 131 pac.) i agregacji per-pacjent: **0.773 balanced accuracy (k-fold)**,
   trafność 0.809. To wartość docelowa i uczciwa (nie 0.89 ze smoke-testu).
2. **Reprezentacja ma znaczenie decydujące:** surowy LSTM się zapada; cechy
   częstotliwościowe przywracają zdolność rozróżniania. Wynik metodyczny wart opisania.
3. **Marker 4–6 Hz nie jest w tym ujęciu samodzielnie wystarczający** — zgodnie z
   wiedzą kliniczną (sama częstotliwość to niedoskonały wyróżnik; liczą się też
   amplituda, regularność, kontekst spoczynek/ruch, asymetria stron).
4. **Heterogeniczność etykiet:** grupa „Parkinson" zawiera podtypy o różnym
   drżeniu — tremor-dominant (13/48), mixed, **akinetic-rigid / hypokinetic-rigid**
   (ci ostatni mają małe/żadne drżenie, więc dla modelu opartego na drżeniu są
   praktycznie nieodróżnialni od zdrowych → szum w etykietach).
5. **Mała próba walidacyjna — zaadresowana k-foldem.** Pojedynczy split per-pacjent
   jest „loterią" (eksp. 7: 0.891 na 5 pac. vs eksp. 8: 0.607 na 26 pac.). Rozwiązanie:
   **walidacja 5-krotna grupowa** — każdy pacjent oceniony raz, wynik jako średnia ±
   rozrzut (eksp. 9). Dalsza praca: rozszerzyć kohortę TD-PD (52) i rozważyć powtarzany
   k-fold (różne ziarna) dla jeszcze węższego przedziału ufności.
6. **Spójność offline↔mobile** pozostaje do ujednolicenia (fs, długość okna,
   filtr zero-phase vs causalny) — patrz `DOKUMENTACJA.md`, ryzyka R2/R3.

---

## 7. Część VI — Zidentyfikowane dźwignie poprawy (kierunek dalszych prac)

Uszeregowane wg oczekiwanego wpływu, z realistycznym celem **~0.75–0.85** (nie 0.99):

1. **Pełna kohorta (469 pacjentów)** — konieczna dla wiarygodności statystycznej
   i (po filtrach) dla wystarczającej liczby właściwych pacjentów.
2. **Filtr podtypu: tylko PD tremor-dominant** — usuwa chorych bez drżenia, którzy
   zaszumiają etykiety. W obecnym podzbiorze: **13 TD-PD** (po pełnej kohorcie
   szacunkowo ~60–70).
3. **Tylko zadania spoczynkowe (Relaxed / RelaxedTask)** — tam marker PD 4–6 Hz
   jest najczystszy; StretchHold (posturalne) go rozmywa. *Uwaga wdrożeniowa niżej.*
4. **Agregacja na poziomie pacjenta z pełnej informacji** — oba nadgarstki
   (w tym **asymetria stron**, typowa dla PD) + wszystkie wybrane zadania → jeden
   wektor cech na pacjenta.
5. **Bogatsze cechy** — regularność/rytmiczność, bezwzględna amplituda (RMS),
   ostrość piku widma, harmoniczne, entropia widmowa.
6. **Lepszy klasyfikator + walidacja k-krotna** — np. drzewa wzmacniane na cechach
   tabelarycznych; k-fold po pacjentach zamiast jednego podziału.

### Decyzje projektowe podjęte 2026-10-20
- **Bez cech demograficznych** — świadomie, by model nie uczył się „stary = PD"
  (ryzyko przecieku i artefaktu); detekcja ma opierać się na sygnale drżenia.
- **Skupienie na drżeniu spoczynkowym** — marker najczystszy, zgodne z pierwotnym
  założeniem badania (pacjenci **tremor-dominant**).
- **Filtr tremor-dominant** — spójny z celem i z redukcją szumu w etykietach.

### Strategia danych — rozważane warianty (2026-10-20)

**(a) Dane syntetyczne (augmentacja) zamiast dociągania pacjentów.**
Rozważone. Werdykt: **przydatne wyłącznie jako augmentacja regularyzująca
w zbiorze treningowym**, nie jako zamiennik realnych pacjentów. Zasady:
- syntetyczne próbki pochodzące od pacjenta X muszą zostać po **tej samej**
  stronie podziału co X — inaczej **wyciek danych** i zawyżone metryki (błąd
  dyskwalifikujący w publikacji);
- walidacja wyłącznie na **realnych** pacjentach — augmentacja nie rozwiązuje
  problemu „tylko 5 realnych pacjentów w walidacji";
- syntetycznych **nie liczymy** jako realne N; oznaczamy je, by móc usunąć.
- Fizycznie sensowne augmentacje dla IMU: skalowanie amplitudy, dodanie szumu,
  małe przesunięcia czasowe, lekki jitter częstotliwości.

**(b) Celowane pobieranie tylko TD-PD (+ zdrowi) — tak, zmniejsza wolumen.**
Sensowne i wprost łagodzi problemy P1/P3/P4 (OOM, łącze). Procedura:
1. Pobrać **wszystkie ~469 etykiet** `patient_*.json` (po ~390 B, łącznie ~0,2 MB
   — trywialne, małe ryzyko OOM).
2. Sparsować `disease_comment` → wyznaczyć ID: TD-PD + zdrowi.
3. Pobrać **tylko sygnały tych pacjentów** i **tylko zadania spoczynkowe** →
   pominięcie ~70% pacjentów i części zadań. Szac. wolumen ~20% pełnego zbioru
   sygnałów tremor.
Zastrzeżenie: potrzebna jest też grupa **zdrowych** (druga klasa); rezygnujemy z
grupy „inne zaburzenia", co jest spójne z zakresem TD-PD vs zdrowi.

### Uwaga wdrożeniowa (ekologiczna trafność — ważne do dyskusji w pracy)
Docelowo pacjent trzyma telefon w dłoni. Kluczowe rozróżnienie:
- **Drżenie spoczynkowe** (ręka rozluźniona, telefon np. na udzie/podparciu) —
  tu PD jest najsilniejsze; marker 4–6 Hz najczystszy. **To zalecana pozycja pomiaru.**
- **Trzymanie telefonu wyciągniętego przed sobą** = pozycja **posturalna**
  (bliżej StretchHold). Drżenie spoczynkowe PD **słabnie** przy posturze, a
  uwidacznia się drżenie samoistne — czyli dla detekcji *PD* gorzej.

**Rekomendacja:** zaprojektować interakcję aplikacji tak, by pomiar odbywał się
w **spoczynku** (np. „połóż rękę z telefonem na kolanie i rozluźnij na 10 s"),
zgodnie z danymi treningowymi Relaxed/RelaxedTask. Wariant `RelaxedTask`
(spoczynek + zadanie poznawcze, które klinicznie *nasila* drżenie spoczynkowe PD)
można odwzorować w aplikacji (prośba o liczenie/odpowiedź w trakcie pomiaru).

---

## 8. Artefakty tej sesji
- Dane: `data/raw/pads_dataset/` (sygnały spoczynkowe pełnej kohorty 131 pac. + 469
  etykiet), `data/raw/pads_csv/`, `data/processed/windows.npz` (3 s),
  `data/processed_5s/` (5 s), `data/processed/windows_td_rest.npz`
  (**pełna kohorta TD-PD vs zdrowi, spoczynek: 6288 okien, 131 pac.**).
- Lista kohorty: `cohort_td_healthy.json` (52 TD-PD + 79 zdrowych, z parsowania
  `disease_comment` wszystkich 469 etykiet).
- Kod: `train_freq.py` (cechy częstotliwościowe + MLP), `eval_patient_level.py`
  (agregacja per-pacjent), **`build_td_rest.py`** (builder zbioru TD-PD vs zdrowi,
  spoczynek — filtr podtypu + zadania + relabel), **`kfold_freq.py`** (walidacja
  k-krotna grupowa po pacjencie).
- Instrukcja dla początkujących: `JAK_URUCHOMIC_TRENING.md` (build / trening /
  k-fold krok po kroku, słowniczek, typowe błędy).
- Modele eksperymentalne: `models/` (3-kl LSTM), `models_2class/`,
  `models_5s_2class/`, `models_freq_2class/`, `models_freq_hvp/`,
  `models_freq_td_rest/` (model docelowy, pełna kohorta).
- Logi: `freq_run.log`, `patient_eval.log`, `freq_hvp.log`,
  `freq_td_rest_full.log`, `kfold_td_rest.log`.
