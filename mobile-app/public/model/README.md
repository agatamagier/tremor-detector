# Model TensorFlow.js — pliki produkcyjne

Aplikacja ładuje model z tego katalogu (`/model/...`) w czasie działania.
Skopiuj tu artefakty wygenerowane przez `ml-pipeline/train.py`:

```
public/model/
├── model.json            # architektura + manifest wag (z tfjs.converters)
├── group1-shard1of1.bin  # wagi (nazwy *.bin mogą się różnić)
└── normalization.json    # { "mean": [...], "std": [...], "channels": [...] }
```

## Skąd je wziąć

1. Po treningu w `ml-pipeline/`:
   - `models/tfjs_model/model.json` + pliki `*.bin` → skopiuj do `public/model/`.
   - `models/normalization.json` → skopiuj do `public/model/`.

   `train.py` zapisuje `normalization.json` automatycznie obok `normalization.npz`.

2. Kolejność kanałów MUSI być zgodna z `preprocess.py`:
   `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`.

## Tryb demo

Bez modelu aplikacja uruchamia się w trybie demo — poziom drżenia ustawia się
suwakiem, co pozwala zobaczyć adaptację UI (48dp → 72dp) bez wytrenowanego modelu.
