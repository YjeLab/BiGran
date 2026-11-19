# BiGran

Official code for **"BiGran: Trustworthy Dual-Granularity Functional Group Identification from FTIR Spectra"** (submitted to *RSC Advances*, under peer review).

BiGran predicts functional groups (17 coarse classes) and bond-vibration types (72 fine classes) from FTIR spectra in a single forward pass, using a shared 89-query decoder with an embedded per-decision reliability head.

This repository contains the **model implementation and released dataset artifacts only**. Training, data-loading and evaluation code are not included.

## Repository contents

```
├── src/bigran/
│   ├── models/       # BiGran-IR: CNN stem → Transformer encoder → physics layer → 89-query decoder
│   ├── physics/      # band anchors, label families, 72→17 label map, label name tables
│   ├── constants.py  # spectral grid constants (500–4100 cm⁻¹, 2 cm⁻¹ step)
│   └── utils/        # minimal model registry
├── data_release/     # dataset split indices and build manifests
├── example.py        # build the model and run a dummy forward pass
└── pyproject.toml
```

## Installation

```bash
pip install -e .
```

## Quick start

```bash
python example.py
```

`example.py` builds BiGran-IR with the released label tables and runs a forward pass on a random spectrum of shape `[B, 2, 1800]`:

```python
from bigran.models import build_bigran

model = build_bigran()
logits = model(x)["logits"]   # [B, 89]: 17 coarse + 72 fine
```

The model expects the second channel to be the first derivative of the absorbance row; see `bigran/models/stem.py` for the physical-kernel layout.

## Datasets

| Benchmark | Spectra | Split protocol |
|---|---|---|
| IRSpectra (Koshelev, *Appl. Spectrosc.*, 2024) | 14,181 | Official random split + molecule-grouped re-split (D1G) |
| NIST-FG-48k (this work) | 48,518 | InChIKey-grouped, molecule-disjoint |

Raw spectra are not redistributed. `data_release/` contains the split indices (spectrum IDs, InChIKeys, split assignments) and build manifests needed to reconstruct every split from the original sources.

## Citation

Citation placeholder — will be updated once the manuscript is published.