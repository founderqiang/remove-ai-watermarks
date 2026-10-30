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
- **Stage 1 (`54` specialist units across `10` canonical providers):**
  - `10` global provider specialists + `44` subfamily and codec/export-regime specialists (`37` active), each combining a block-diagonal MLP Soup, a vectorized fixed-depth `HistGradientBoosting` tree ensemble, and a distilled `ExtraTrees -> MLP` student with calibration-reoptimized distilled blend weights (`w_e <= 0.25`), pooled via drift-bounded zero-inflation subfamily blending across 16 real-image varieties, refined by an unknown-domain debiased Stage-1.5 zero-inflated hinge meta-router (`router_meta_tau`, `router_meta_w`), and routed by a unified damped competitive margin (`gamma = 0.60`).
- **Stage 2 (`9` stacked version heads, `1010-d`):**
  - Feature-subsampled vectorized `HistGradientBoosting` tree ensembles (including a multi-depth forest soup on `flux`) taking the `956-d` features (and `54` Stage-1 unit log-odds scores) to predict the canonical model/export subfamily (`version_subfamily`).
- **Artifact:** `multi-provider-source-v3.npz` (`allow_pickle=False`)
- **SHA-256:** `b23ef046da3cb85b65e5985a276cbf2912022d79a1837f5d0967631f7c04598e`

The artifact contains only learned parameters, normalization constants, tree arrays, labels, and calibrated thresholds. It does not contain training images, paths, image hashes, embeddings, or a training catalog.

## Evaluation (`56,611` clean audited samples; `10,586` locked test)

The operating point was selected on a disjoint `8,365`-image calibration split across 16 fine-grained real-image varieties and group-locked (`parent_sha256`) native, light-JPEG, and heavy-JPEG/resize views. On the `10,586`-image locked test split (`4,539` multi-variety `unknown` negatives + `6,047` AI provider images including preprocessed social/JPEG derivatives):

- **Unified 10-provider router:** `99.41%` `unknown` specificity (`0.59%` test `unknown` FPR; `0.87%` photographic FPR and `0.34%` non-photographic structured FPR across all 15 non-photo varieties), `49.43%` aggregate target recall across native and recompressed test views (`96.67%` provider accuracy when firing), and `0.9419` binary ROC-AUC / `0.9579` PR-AUC on the monotonic `confidence` score (`_sigmoid_scalar(excess[best_cls])`).

| Provider | Test count | Standalone ROC AUC | Router recall | Router precision | Stage-2 version accuracy |
|---|---:|---:|---:|---:|---:|
| `openai` | 1,478 | 0.9657 | 60.8% | 96.6% | 85.0% |
| `google` | 1,382 | 0.9661 | 64.1% | 96.9% | 92.6% |
| `bytedance` | 389 | 0.9445 | 53.2% | 97.6% | 90.0% |
| `flux` | 1,357 | 0.9234 | 34.2% | 94.9% | 85.8% |
| `stable_diffusion` | 955 | 0.9582 | 37.6% | 97.0% | 85.9% |
| `meta` | 94 | 0.9767 | 78.7% | 91.4% | — (`meta:imagine`) |
| `midjourney` | 123 | 0.9215 | 36.6% | 81.8% | 87.8% |
| `ideogram` | 118 | 0.8868 | 10.2% | 70.6% | 64.4% |
| `firefly` | 103 | 0.9170 | 33.0% | 87.2% | 100.0% |
| `xai` | 48 | 0.9283 | 20.8% | 83.3% | 72.9% |

### Held-out real-image specificity by variety (`4,539` test negatives across 16 real classes + foreign AI)

| Real-image variety | Test count | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 2,183 | 19 | 0.87% | 99.13% |
| `photo_retouched` | 339 | 1 | 0.29% | 99.71% |
| `classical_inpaint` | 17 | 0 | 0.00% | 100.00% |
| `digital_art` | 441 | 1 | 0.23% | 99.77% |
| `game_art` | 174 | 1 | 0.57% | 99.43% |
| `vector` | 135 | 1 | 0.74% | 99.26% |
| `interface` | 180 | 0 | 0.00% | 100.00% |
| `diagram` | 98 | 1 | 1.02% | 98.98% |
| `chart_data_viz` | 159 | 0 | 0.00% | 100.00% |
| `technical_architecture` | 84 | 0 | 0.00% | 100.00% |
| `procedural_graphic` | 13 | 0 | 0.00% | 100.00% |
| `document` | 165 | 0 | 0.00% | 100.00% |
| `gov_scientific_doc` | 32 | 0 | 0.00% | 100.00% |
| `receipt_invoice` | 181 | 1 | 0.55% | 99.45% |
| `print_publishing` | 229 | 1 | 0.44% | 99.56% |
| `stress_recompressed` | 103 | 1 | 0.97% | 99.03% |
| `foreign_ai` | 6 | 0 | 0.00% | 100.00% |

### External blind multi-variety real-image verification (`selective-v8`, `2,267` untouched images)

Evaluated on the hash-disjoint external `selective-v8` blind real-image cohorts (`2,267` images never seen in train, calibration, or test): **10 / 2,267 false positives (`0.44%` overall FPR, `99.56%` specificity)**.

| Blind `selective-v8` cell | Evaluated | False positives | False-positive rate | Specificity |
|---|---:|---:|---:|---:|
| `photo` | 450 | 5 | 1.11% | 98.89% |
| `vector` | 332 | 0 | 0.00% | 100.00% |
| `game_art` | 230 | 5 | 2.17% | 97.83% |
| `interface` | 342 | 0 | 0.00% | 100.00% |
| `diagram` | 305 | 0 | 0.00% | 100.00% |
| `document` | 308 | 0 | 0.00% | 100.00% |
| `stress_recompressed` | 300 | 0 | 0.00% | 100.00% |

## Intended use

Install `remove-ai-watermarks[source-classify]` and call the explicit Python API `classify_source(path)` or CLI command `remove-ai-watermarks classify-source <IMAGE>`. `identify` does not run this model. Set `RAIW_SOURCE_CLASSIFY_WEIGHTS` to the `.npz` file or its parent directory for offline deployments.
