---
license: apache-2.0
library_name: numpy
pipeline_tag: image-classification
tags:
  - image-forensics
  - source-attribution
  - open-set-classification
---

# RAIW multi-provider source-pipeline classifier (`multi-provider-source-v3`)

This repository contains the pickle-free (`allow_pickle=False`) derivative weights for the abstaining `remove-ai-watermarks` pixel source-pipeline classifier across **10 AI provider families** (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`, `meta`, `midjourney`, `ideogram`, `firefly`, `xai`) plus `unknown` (`classify_source` / `remove-ai-watermarks classify-source`).

This model is **not a SynthID detector** and does not decode or verify any cryptographic or invisible watermark payload (`watermark_truth="unknown"`). Its labels describe complete generation and export pixel pipelines, so renderer, geometry, encoding, post-processing, and watermark state may be confounded. `unknown` does not mean that SynthID is absent, and a provider label does not prove that SynthID is present.

## Model architecture

- **Input (`956-d`):**
  - `124-d` spatial opponent-residual and gradient forensic features (`extract_124d`).
  - `768-d` phase-free `16x16` log-power and radial spectral features (`source_spectral`).
  - `64-d` directional / block-boundary (`8x8` and `16x16`) / chroma-aliasing / azimuthal-anisotropy features (`augment_892d_to_956d`).
- **Stage 1 (`43` specialist units across `10` canonical providers):**
  - `10` global provider specialists + `33` subfamily specialists, each combining a block-diagonal MLP Soup, a vectorized fixed-depth `HistGradientBoosting` tree ensemble, and a distilled `ExtraTrees -> MLP` student with calibration-reoptimized distilled blend weights (`w_e <= 0.25`), pooled via drift-bounded zero-inflation subfamily blending across 16 real-image varieties and routed by a unified damped competitive margin (`gamma = 0.60`).
- **Stage 2 (`9` stacked version heads, `999-d`):**
  - Feature-subsampled vectorized `HistGradientBoosting` tree ensembles taking the `956-d` features (and Stage-1 log-odds scores) to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v3.npz` (`allow_pickle=False`)
- **SHA-256:** `f7f5da50edccc264f3a326208abc4fc6f36b7e085f0fc0507dff4b4f260d21b3`

The artifact contains only learned parameters, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation (`40,057` clean audited samples; `8,282` locked test)

The operating point was selected on a disjoint `6,595`-image calibration split across 16 fine-grained real-image varieties. On the `8,282`-image locked test split (`4,480` multi-variety `unknown` negatives + `3,802` AI provider images):

- **Unified 10-provider router:** `99.78%` `unknown` specificity (`0.22%` test `unknown` FPR; `0.46%` photographic FPR and `0.00%` non-photographic structured FPR across all 15 non-photo varieties) and `56.79%` aggregate target recall.

| Provider | Test count | Standalone ROC AUC | Router recall | Router precision | Stage-2 version accuracy |
|---|---:|---:|---:|---:|---:|
| `openai` | 1,146 | 0.9833 | 65.2% | 97.1% | 87.6% |
| `google` | 1,038 | 0.9816 | 62.3% | 97.4% | 95.0% |
| `bytedance` | 371 | 0.9455 | 50.1% | 98.9% | 90.0% |
| `flux` | 740 | 0.9641 | 43.4% | 96.4% | 92.8% |
| `stable_diffusion` | 133 | 0.9877 | 57.1% | 88.4% | 91.7% |
| `meta` | 94 | 0.9758 | 78.7% | 92.5% | — (`meta:imagine`) |
| `midjourney` | 77 | 0.9751 | 67.5% | 70.3% | 96.1% |
| `ideogram` | 72 | 0.9809 | 40.3% | 72.5% | 61.1% |
| `firefly` | 83 | 0.9066 | 22.9% | 100.0% | 100.0% |
| `xai` | 48 | 0.9247 | 16.7% | 72.7% | 68.8% |

### Held-out real-image specificity by variety (`4,480` test negatives across 16 real classes + foreign AI)

| Real-image variety | Test count | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 2,183 | 10 | 0.46% | 99.54% |
| `photo_retouched` | 339 | 0 | 0.00% | 100.00% |
| `classical_inpaint` | 17 | 0 | 0.00% | 100.00% |
| `digital_art` | 441 | 0 | 0.00% | 100.00% |
| `game_art` | 174 | 0 | 0.00% | 100.00% |
| `vector` | 135 | 0 | 0.00% | 100.00% |
| `interface` | 180 | 0 | 0.00% | 100.00% |
| `diagram` | 98 | 0 | 0.00% | 100.00% |
| `chart_data_viz` | 159 | 0 | 0.00% | 100.00% |
| `technical_architecture` | 84 | 0 | 0.00% | 100.00% |
| `procedural_graphic` | 13 | 0 | 0.00% | 100.00% |
| `document` | 165 | 0 | 0.00% | 100.00% |
| `gov_scientific_doc` | 32 | 0 | 0.00% | 100.00% |
| `receipt_invoice` | 122 | 0 | 0.00% | 100.00% |
| `print_publishing` | 229 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 103 | 0 | 0.00% | 100.00% |
| `foreign_ai` | 6 | 0 | 0.00% | 100.00% |

### External blind multi-variety real-image verification (`selective-v8`, `2,267` untouched images)

Evaluated on the hash-disjoint external `selective-v8` blind real-image cohorts (`2,267` images never seen in train, calibration, or test): **3 / 2,267 false positives (`0.13%` overall FPR, `99.87%` specificity)**.

| Blind `selective-v8` cell | Evaluated | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 450 | 2 | 0.44% | 99.56% |
| `vector` | 332 | 0 | 0.00% | 100.00% |
| `game_art` | 230 | 1 | 0.43% | 99.57% |
| `interface` | 342 | 0 | 0.00% | 100.00% |
| `diagram` | 305 | 0 | 0.00% | 100.00% |
| `document` | 308 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 300 | 0 | 0.00% | 100.00% |

## Intended use

Install `remove-ai-watermarks[source-classify]` and call the explicit Python API `classify_source(path)` or CLI command `remove-ai-watermarks classify-source <IMAGE>`. `identify` does not run this model. Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file or its parent directory for offline deployments.
