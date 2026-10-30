# Chroma1 engine research (2026-08-29/30)

> Metric correction: the OCR metric from `scripts/fidelity_metrics.py` is
> normalized edit distance (NED), edit distance divided by the longer
> normalized string. The historical values are unchanged; the earlier CER
> label incorrectly implied normalization by reference length.

> Research archive. This page records experiments and decisions from the dates
> above. Current package behavior is defined by the
> [supported signals](supported-signals.md), [known limitations](known-limitations.md),
> and [module internals](module-internals.md). Nothing here is shipped.

Correction 2026-09-23: the claim below is out of date. diffusers now ships
`Flux2KleinInpaintPipeline`, whose `strength` argument drives `get_timesteps`
and `scheduler.scale_noise` (checked against the installed 0.40.0 source), and FLUX.2 klein 4B
is Apache-2.0. An all-white mask does not make it plain img2img, though: see
[FLUX.2 klein as a global stage](#flux2-klein-as-a-global-stage-2026-09-27).

Cited experiment behind issue #88 ("Please integrate Flux 2"): FLUX.2 has no
native strength img2img anywhere (BFL reference code or diffusers; its image
inputs are reference conditioning and outputs start from pure noise), so the
requestable scrub lane in the FLUX family is **Chroma1** -- the FLUX.1
architecture re-trained on open data, Apache-2.0 on weights, with a native
`ChromaImg2ImgPipeline` strength parameter. This page records what was measured
before any integration decision.

## Setup

All Chroma1 runs: `lodestones/Chroma1-HD` bf16 on Modal H100, seed 0, the
neutral scrub prompt (`high quality, sharp, detailed, faithful to the original`
/ `blurry, lowres, distorted text, garbled text, artifacts`), guidance 5.0, and
**step compensation**: the calibrated request is `ceil(4 / strength)`.
Chroma rounds the start index rather than the active step count: this runs
five effective steps at strengths 0.09 and 0.17, and four at 0.20, 0.125,
and 0.40. The nominal count is four, not a promise of four effective steps.
The calibrated requests remain unchanged.
Without it, strength 0.05 runs a single step and the floor reads artificially
high (the first, uncompensated sweep put the OpenAI boundary at (0.05, 0.10];
compensated it is (0.05, 0.06]).

Two Flux-family traps found while wiring the prototype:

- without explicit `width`/`height`, the img2img pipeline silently generates
  1024x1024 regardless of input aspect;
- dimensions floor to the /16 latent-patch grid, so the result must be resized
  back to the input size for pixel-exact comparison.

Harnesses (Modal GPU scripts kept outside this repository; each mounts this
library's `src/` and reads its tracked corpora): `chroma_scrub_prototype.py`
(first sweep), `engine_quality_price_probe.py` (head-to-head with the
qwen-zimage path), `engine_cohort_calibration.py` (the cohort ladders below),
`chroma_meta_high_rungs.py` (the Meta rungs above 0.08),
`chroma_muse_expansion.py` (harvest-2 Meta API fixtures),
`chroma_openai_expansion.py` (extra OpenAI SynthID carriers),
`chroma_seed_probes.py` (seed-1/2 at measured first-cleans).
These harnesses are internal tooling and are not distributed, so a reader outside
the project cannot rerun them. Generated outputs stay outside the repository
(`out/`, gitignored); the durable verdicts are the tables on this page.

## Measured floors

Oracle per cohort: OpenAI via openai.com/verify (SynthID read separately from
C2PA); Meta via the anonymous meta.ai/identification oracle. Each fixture's
rungs were checked ascending and stopped at the first clean verdict.

| cohort | fixture | detected at | clean from |
|---|---|---|---|
| OpenAI | typography card 1122x1402 | 0.05 | 0.06 |
| OpenAI | 9-face grid 1448x1086 | 0.06 | 0.075 |
| OpenAI | full-pipeline quality fixture 1122x1402 | 0.05 | 0.06 |
| Meta | gen_lighthouse_watercolor 1600x1600 | 0.08 | 0.10 |
| Meta | gen_fox_forest 1600x1600 | 0.06 | 0.08 |
| Meta | gen_night_city 1920x1280 | 0.045 | 0.06 |
| Meta | gen_studio_mug 1600x1600 | 0.03 | 0.045 |
| Meta | gen_text_poster 1600x1600 | -- | <=0.03 |

Derived operating points from this initial calibration (the OpenAI value was
superseded by the 2026-09-07 holdout below):

- **OpenAI: 0.09, superseded.** (0.075 + (0.075 - 0.06)). Against qwen-zimage 0.07675 and
  sdxl-zimage 0.15. The typography boundary (0.05, 0.06] is seed-stable across
  seeds 0, 1, 2.
- **Meta: 0.17** (0.10 + (0.10 - 0.03)), oracle-verified clean on the worst
  fixture at 0.17. Against qwen-zimage 0.1. Chroma needs ~1.7x qwen's strength
  on Content Seal.
- **Google: 0.40, measured 2026-08-30 across all four fixtures.** A
  metadata-stripped original verified as pixel-DETECTED through the @synthid
  Verify AI agent (the C2PA-carrying original answers about C2PA instead,
  which is not a pixel verdict).

  | fixture | detected at | clean from |
  |---|---|---|
  | Gemini_Generated_Image_633uuy | 0.20 | 0.25 |
  | Gemini_Generated_Image_akdbei | 0.20 | 0.25 |
  | Gemini_Generated_Image_y48j3c | 0.08 | 0.12 |
  | Gemini_Generated_Image_3mc4t9 | 0.08 | 0.12 |

  Derived by the shipped rule: worst first-clean 0.25 + spread (0.25 - 0.12
  = 0.13) = 0.38, rounded up to the measured rung **0.40**, then
  oracle-verified clean on the worst fixture. Against qwen-zimage 0.27:
  Chroma needs ~1.5x the strength on Google SynthID. At 0.15-0.20 the tool
  still reads positive while its wording degrades from "most or all" to
  "part of", so the boundary band is real signal decay, not noise.

  Oracle mechanics learned while bracketing: the quota is per ACCOUNT and
  additionally dedupes SIMILAR images ("quota for checking images similar to
  this one"), roughly three checks per account window; switching accounts
  inside one browser works. A fresh Playwright context with the same Google
  session cookies (exported browser-wide via CDP `Storage.getCookies`, never
  through the transcript) carries working sign-in and its own check budget.
  The agent's own instructions leak into the answer on some renders and
  include an abstention rule for text-dominated images - always read the TOOL
  outcome, never the prose.
- **Microsoft: 0.125**, measured 2026-08-30 on the SAME three valid Paint
  carriers behind the qwen floor (staged under
  `out/cohort-calibration/microsoft/` with pixel-identical metadata-stripped
  controls; they are not publication-cleared and are never committed).
  Controls re-verified `Microsoft AI detected` on the day of the run. Against
  qwen-zimage 0.15: Chroma needs LESS strength on InvisMark.

  | source (Paint) | detected at | clean from | qwen first-clean |
  |---|---|---|---|
  | paint-1 (47d87e99, 1536x1024) | 0.06 | 0.08 | 0.055 |
  | paint-2 (a579fb70, 1536x1024) | -- | <=0.04 | 0.04125 |
  | paint-3 (883191b5, worst qwen source) | 0.06 | 0.08 | 0.095 |

  Derived by the shipped rule: 0.08 + (0.08 - 0.04) = 0.12, rounded to the
  measured rung **0.125**, which was then oracle-verified clean on both worst
  sources. Oracle: the public `ai.azure.com/nextgen/validate` page; its
  `Inconclusive` verdict is the site's clean outcome and provides the provider
  oracle used for this certification. The API exposes a more detailed report,
  not a different class of certification.

## Fidelity at the floors (scripts/fidelity_metrics.py)

qwen-zimage rows ran through the package's own `QwenZImagePipeline` on the same
H100 (production path, both stacks resident, prompt cache warm).

OpenAI typography (chroma 0.06 vs qwen 0.07675):

| | NED | img LPIPS | SSIM | PSNR |
|---|---|---|---|---|
| qwen-zimage | 0.241 | 0.083 | 0.834 | 27.7 |
| chroma1 | **0.138** | **0.049** | **0.881** | **31.2** |

OpenAI full-pipeline fixture (chroma 0.09 vs qwen 0.07675): chroma wins every
metric (NED 0.155 vs 0.259, LPIPS 0.056 vs 0.094, PSNR 29.6 vs 26.1).

OpenAI 9-face grid (chroma 0.075 global-only vs qwen 0.07675 with its Z-Image
face stage):

| | ID cos | face LPIPS | img LPIPS | SSIM | PSNR |
|---|---|---|---|---|---|
| qwen-zimage | **0.857** | **0.051** | 0.065 | 0.865 | 31.1 |
| chroma1 (no face stage) | 0.788 | 0.067 | **0.060** | **0.890** | **32.4** |

The face stage regenerates from the ORIGINAL crops and is engine-independent,
so an integrated `chroma-zimage` would composite the same faces over the
chroma global; whether that closes the identity gap is an integration-time
measurement, not a given.

Meta at the respective floors (chroma 0.17 vs qwen 0.1): qwen wins perceptual
fidelity on textured content (LPIPS: fox 0.163 vs 0.211, lighthouse 0.173 vs
0.248, night city 0.187 vs 0.277), chroma wins the text poster decisively
(LPIPS 0.008 vs 0.012, NED 0.000 vs 0.021) and the studio mug is a split. The
extra 0.07 of strength costs real fidelity on photoreal content.

Google at the respective floors (chroma 0.40 vs qwen 0.27, qwen row includes
its Z-Image face stage):

| fixture | qwen LPIPS/SSIM/ID cos | chroma LPIPS/SSIM/ID cos |
|---|---|---|
| 633uuy | **0.405 / 0.577** / - | 0.499 / 0.576 / - (NED 1.000 vs 0.333) |
| akdbei | **0.440 / 0.552** / - | 0.540 / 0.557 / - (NED 0.796 vs 0.367) |
| y48j3c (18 faces) | **0.436 / 0.608 / 0.801** | 0.560 / 0.548 / 0.279 |
| 3mc4t9 (5 faces) | **0.422 / 0.529 / 0.921** | 0.562 / 0.461 / 0.143 |

qwen wins every axis that matters on every fixture; at 0.40 the Chroma global
regeneration itself collapses face identity (0.14-0.28 cosine) and destroys
dense text (NED up to 1.0). The extra 0.13 of strength is simply too much
regeneration. This is the Meta pattern amplified.

Microsoft at the respective floors (chroma 0.125 vs qwen 0.15): chroma wins
EVERY metric on EVERY source while scrubbing at lower strength --

| source | NED q/ch | img LPIPS q/ch | SSIM q/ch | PSNR q/ch |
|---|---|---|---|---|
| paint-1 | 0.741 / 0.241 | 0.164 / 0.106 | 0.742 / 0.847 | 24.8 / 29.3 |
| paint-2 | 0.826 / 0.391 | 0.253 / 0.108 | 0.536 / 0.633 | 22.5 / 25.5 |
| paint-3 | 0.000 / 0.000 | 0.038 / 0.020 | 0.762 / 0.953 | 36.3 / 42.2 |

The Microsoft cohort therefore behaves like the OpenAI cohort, not like Meta:
for InvisMark carriers Chroma1 is the better engine at a lower operating point.

## Price (warm H100 container)

| | qwen-zimage | chroma1 |
|---|---|---|
| seconds per image | 2.44 | 3.62 |
| VRAM peak | 31.9 GiB | 29.2 GiB |
| USD per 1000 images ($3.95/h) | $2.68 | $3.97 |
| weights in cache | ~70 GB | ~28 GB |

The chroma number includes a T5 encode per call; the shipped qwen path caches
prompt embeddings, and the same treatment would put chroma near 1.5-2 s/image.
Both stacks fit an 80 GB card; chroma's smaller footprint likely fits smaller
cards untested.

## Matched-strength comparison: the floors hide the engine (2026-08-30 addendum)

The floor-based tables above compare chroma at its DERIVED floor (worst case
+ spread) against qwen at its shipped floor - different perturbation levels.
At the strength where each fixture ACTUALLY turns clean (chroma first-clean
rungs, all oracle-verified above), the picture reverses:

Google (chroma@first-clean vs qwen@0.27):

| fixture | chroma rung | LPIPS q/ch | SSIM q/ch | PSNR q/ch | NED q/ch | ID q/ch |
|---|---|---|---|---|---|---|
| 633uuy | 0.25 | 0.405/**0.375** | 0.577/**0.673** | 18.8/**22.3** | **0.333**/0.667 | - |
| akdbei | 0.25 | 0.440/**0.409** | 0.552/**0.650** | 18.6/**22.5** | 0.367/**0.163** | - |
| y48j3c | 0.12 | 0.436/**0.329** | 0.608/**0.700** | 21.0/**25.8** | 1.0/1.0 | **0.801**/0.611 |
| 3mc4t9 | 0.12 | 0.422/**0.318** | 0.529/**0.644** | 17.7/**21.3** | 0.443/**0.229** | **0.921**/0.410 |

Meta (chroma@first-clean vs qwen@0.1):

| fixture | chroma rung | LPIPS q/ch | SSIM q/ch | PSNR q/ch | NED q/ch |
|---|---|---|---|---|---|
| lighthouse | 0.10 (equal) | 0.173/**0.097** | 0.783/**0.886** | 25.6/**30.8** | - |
| fox | 0.08 | 0.163/**0.083** | 0.778/**0.907** | 25.6/**31.4** | - |
| night_city | 0.06 | 0.187/**0.092** | 0.748/**0.893** | 23.9/**29.6** | 0.850/**0.250** |
| studio_mug | 0.045 | 0.024/**0.009** | 0.984/**0.992** | **39.0**/37.8 | - |
| text_poster | 0.03 | 0.012/**0.002** | 0.984/**0.994** | 34.0/**41.2** | 0.021/**0.007** |

At effective-clean strength Chroma1 wins whole-image fidelity on essentially
every fixture (often by 4-6 dB) and text on five of six; qwen's remaining
edge is face identity, which its Z-Image face stage buys (0.80-0.92 vs
0.41-0.61 cosine) and a chroma-zimage profile would inherit on top of its own
global pass.

**Why the floors still matter:** a shipped strength must clear the worst
unseen content, and Chroma1's per-fixture boundaries scatter much wider
(Google 0.12-0.25, Meta 0.03-0.10) than the spread the flat floors were built
to absorb - so its derived floors (0.40 / 0.17) pay a large margin penalty
and at those floors the extra strength genuinely destroys quality. The
engines differ less in "quality per strength" than in how predictable their
boundary is. The integration lever this exposes: a content-adaptive strength
policy for chroma-zimage (the project already ships resolution-adaptive
denoise for qwen) could capture chroma's low per-content boundaries; the two
hard Google fixtures are both dense-text cards and the easy ones are
photoreal, so a content-class signal may predict the rung.

## Pre-ship validation (2026-08-30)

With the `chroma-zimage` profile implemented
(`_internal/chroma_zimage_pipeline.py` on `TwoStageZImagePipeline`, floors in
`watermark_profiles`), two ship-gate checks were run:

1. **Seed stability at the worst Meta boundary.** lighthouse @ 0.10 with
   seeds 1 and 2: both CLEAN on the meta.ai oracle. The Meta boundary is not a
   seed-0 fluke. The Google seed probes (633uuy @ 0.25) delivered a critical
   finding: **seed 0 was CLEAN but seed 1 was DETECTED** -- the 633uuy
   boundary is seed-dependent, which validates the flat 0.40 floor's 0.15
   margin exactly the way the qwen SDXL precedent did (the same image at the
   same strength passing or failing run-to-run near the threshold). Seed 2
   of `seedprobe_google_633uuy_s0.25_seed2.png` was checked twice on
   2026-08-31 after a human drag-drop. Work Gemini (u/2, Pro,
   `https://gemini.google.com/u/2/app/ebc7c19d22f85bb9`) with the
   ordinary question "Is there a SynthID watermark in this image?"
   auto-titled "SynthID Watermark Detection Results" and returned "No
   reliable signals were detected indicating how the content was created,
   and you should look for other supporting evidence from trusted
   sources." Plus `@synthid` (u/0,
   `https://gemini.google.com/u/0/app/658b1470307811b7`) with the built-in
   verification question auto-titled "SynthID Image Verification Results"
   and returned "No digital watermark from Google AI was detected in this
   image, indicating that it was not created or edited using Google's AI
   models." Both are CLEAN. An earlier Plus `@synthid` attempt on the
   same file hung on "Connecting to Verify AI" with auto-title "SynthID
   Image Verification Failed"; Flash prose abstained because "the tool
   response does not confirm" Google AI, which is not a tool verdict.
   Workspace Gemini's More-uploads menu does not list Photos, but a
   human drag-drop of a local PNG does attach; one Workspace account in
   the same Chrome has no Gemini app. Seed 0 CLEAN / seed 1 DETECTED /
   seed 2 CLEAN at 0.25
   leaves the 633uuy boundary seed-dependent on the Plus `@synthid`
   path; the 0.40 floor is already established by the seed-1 flip.
2. **Full-path face-stage interaction.** The complete profile run (Chroma1
   global at the shipped Google floor 0.40, then the inherited
   YuNet/SAM/Z-Image face stage) on the 5-face fixture 3mc4t9: CLEAN on the
   @synthid oracle. The composited face regions do not re-introduce a
   detectable signal on top of the floor.

Generation harness: `chroma_preship_validation.py` (kept outside this
repository); outputs under
`out/preship-chroma/` (gitignored).

## Verdict

Engine quality is cohort-dependent, and after measuring all four cohorts the
split is even and sharp:

- On the initial OpenAI fixtures Chroma1 was better at the then-current floor,
  but the later holdout moved that floor to 0.1375 and changed the router decision.
- On Microsoft InvisMark content Chroma1 is better at every measured fidelity
  axis at a LOWER floor than qwen (0.125 vs 0.15).
- On Meta Content Seal Chroma1 needs 1.7x the strength (0.17 vs 0.1) and loses
  LPIPS on textured content at its floor.
- On Google SynthID Chroma1 needs ~1.5x the strength (0.40 vs 0.27) and loses
  every fidelity axis on every fixture, with face identity collapsing at 0.40
  (0.14-0.28 cosine vs qwen's 0.80-0.92).

A single-engine `chroma-zimage` default at a flat per-vendor floor is NOT
supported: Google and Meta floors land too high and the quality at those
floors is worse than qwen's. But the matched-strength addendum changes what
IS supported: at the strength each image actually needs, Chroma1 regenerates
better than qwen almost everywhere except face identity (which the shared
face stage would supply). The two honest integration options:

1. Per-cohort engine router at flat floors: chroma for OpenAI/Microsoft,
   qwen for Google/Meta - conservative, ships on the measurements as they
   stand.
2. `chroma-zimage` with a content-adaptive strength policy that predicts the
   per-image rung - potentially better everywhere, but the predictor is new
   research (the measured signal so far: dense-text content needs the high
   rungs, photoreal the low ones) and needs its own oracle validation before
   any ship.

Both need the seed-sensitivity and face-stage composition checks called out
above before shipping. Both checks have now been run (Pre-ship validation
above): the Meta boundary is seed-stable across seeds 0-2 and the full
profile path stays clean through the face stage on the shipped Google floor.
The profile ships with option 1's floors plus a content-adaptive Google arm
(face content resolves to 0.125 instead of 0.40, keyed on YuNet detection).
Option 2's full adaptive policy for the remaining cohorts remains the
documented follow-up. The Meta arm was measured on 2026-08-30/31 and
does not have a shippable split; see the expansion section below.

## Meta content-adaptive expansion (2026-08-30/31)

Goal: test whether `flat_ratio` (fraction of 16x16 blocks with luma std < 8)
predicts Chroma1's first-clean Content Seal boundary well enough to ship a
Meta arm analogous to Google's face-count split.

**Harvest 1, from a 495-file uncommitted corpus scan, was not a diverse expansion.**
Seven "unique" files were staged under `out/cohort-calibration/meta/muse-*`
(gitignored). After control checks on `meta.ai/identification`:

| fixture | control | first-clean | note |
|---|---|---|---|
| muse-1 | CLEAN | -- | seal did not survive the metadata-stripped control |
| muse-2 | DETECTED | (0.08, 0.10] | photoreal portrait; 0.06 and 0.08 DETECTED, 0.10 CLEAN |
| muse-3 | DETECTED | (0.015, 0.03] | photoreal portrait, different subject |
| muse-4 | DETECTED | (0.08, 0.10] | same subject as muse-2/6/7; 0.06 and 0.08 DETECTED, 0.10 CLEAN |
| muse-5 | -- | -- | byte-identical to committed `gen_lighthouse_watercolor.webp` |
| muse-6 | DETECTED | (0.06, 0.08] | same subject as muse-4; 0.06 DETECTED, 0.08 CLEAN |
| muse-7 | DETECTED | (0.045, 0.06] | same subject as muse-2/4/6; 0.045 DETECTED, 0.06 CLEAN |

Five valid new points, four of them near-duplicate portraits of one
subject, with first-clean already spanning 0.06-0.10 **inside that
subject**. Google's shipped split had zero overlap between classes and
identical boundaries inside a class. This harvest cannot support a
predictor.

`flat_ratio` on the original five plus the new portraits is not monotonic
with first-clean either (studio_mug 0.973 / 0.045 easy; lighthouse 0.602 /
0.10 hard; muse-4 0.789 / 0.10 hard). OpenAI stays unshippable for the
same reason as before: one text fixture vs one face fixture, and the face
fixture was the harder one, the opposite of Google's split.

**Harvest 2** takes six files from the 2026-08-26 Muse Image API corpus
(61 independent `muse-image-1.0` generations, no hash overlap with the
committed five), chosen to span class and `flat_ratio`:

| fixture | class | flat_ratio | edge_density |
|---|---|---|---|
| architecture_mosque | dense architecture | 0.037 | 0.367 |
| product_sneaker | product / high-flat | 0.874 | 0.018 |
| text_poster_bakery | text | 0.368 | 0.191 |
| portrait_weathered_fisherman | photoreal portrait | 0.470 | 0.108 |
| illustration_watercolor_botanical | illustration | 0.442 | 0.089 |
| scene_tokyo_alley | busy scene | 0.258 | 0.175 |

Harness: `chroma_muse_expansion.py` (kept outside this repository). Outputs under
`out/cohort-calibration/meta-api/` (gitignored). Chroma1 ladders (seven
rungs, seed 0, nominal four-step compensated requests) ran for all six. Oracle
2026-08-31, anonymous `playwright-isolated` against
`meta.ai/identification`. All six controls DETECTED (valid carriers).

| fixture | flat_ratio | detected at | clean from |
|---|---|---|---|
| architecture_mosque | 0.037 | 0.08 | 0.10 |
| illustration_watercolor_botanical | 0.442 | 0.08 | 0.10 |
| scene_tokyo_alley | 0.258 | 0.06 | 0.08 |
| text_poster_bakery | 0.368 | 0.045 | 0.06 |
| portrait_weathered_fisherman | 0.470 | 0.045 | 0.06 |
| product_sneaker | 0.874 | 0.03 | 0.045 |

The worst first-clean is still 0.10, the same as the committed
lighthouse, so the shipped Meta floor 0.17 does not move.

**No Meta adaptive arm.** Google's shipped split had zero overlap
between classes and identical first-cleans inside a class. Combined
with the original five, Meta first-cleans are a continuum from 0.03 to
0.10, and `flat_ratio` does not separate them:

- the 0.10 cluster is mosque 0.037, botanical 0.442, and lighthouse
  0.602;
- the 0.045 cluster is sneaker 0.874 and studio_mug 0.973;
- harvest-1 photoreal portraits were high-flat (0.75-0.85) AND hard
  (0.08-0.10), so a high-flat easy-arm would misroute them.

A content-class rule fails the same way: two text posters first-clean
at 0.03 and 0.06, and two watercolor-like images (lighthouse, botanical)
share 0.10 with dense architecture.

The leftover rungs were finished in a 2026-08-31 evening UTC
anonymous isolated session (fresh-navigation, wait for "Upload
another file"; all ten `POST /api/ai-detector` calls returned 200):

| check | verdict |
|---|---|
| muse-2 @ 0.08 | DETECTED |
| muse-3 @ 0.015 / 0.03 / 0.045 | DETECTED / CLEAN / CLEAN |
| muse-7 @ 0.045 / 0.06 | DETECTED / CLEAN |
| mosque @ 0.10 seeds 1, 2 | CLEAN / CLEAN |
| botanical @ 0.10 seeds 1, 2 | DETECTED / DETECTED |

botanical's seed-0 first-clean of 0.10 is not seed-stable;
lighthouse @ 0.10 seeds 1 and 2 were CLEAN. That is the same class
of run-to-run flip the 0.17 floor's margin exists for. No
Google-style binary split: harvest-1 first-cleans now span 0.03
(muse-3) through 0.10 (muse-2/4), and 0.06-0.10 inside one subject
(muse-7 / muse-6 / muse-2).

## OpenAI content-adaptive expansion (2026-08-31)

Goal: test whether face_count or `flat_ratio` predicts Chroma1's OpenAI
SynthID first-clean the way YuNet does for Google.

Committed three (YuNet + `flat_ratio` on the files, first-cleans from the
2026-08-29/30 web oracle):

| fixture | faces | flat_ratio | first-clean |
|---|---|---|---|
| typography 02_03_55 | 0 | 0.642 | 0.06 |
| quality 02_02_23 | 0 | 0.695 | 0.06 |
| 9-face grid 05_30 | 9 | 0.473 | 0.075 |

Face content is the harder of the two classes, the opposite of Google.
Two zero-face cards share a boundary, which is a hint, not a split.

**Images API is not a carrier source.** `gpt-image-1` and `gpt-image-1.5`
generations (medium, 1024) carry C2PA (`detected`) but SynthID
`not_detected` on `POST /v1/content_provenance_checks`. Contrastive-pair
"openai" recreations are also SynthID-negative. New ChatGPT-UI downloads
are not in the tree beyond the three committed files.

**Uncommitted corpus scan (one day, 2026-07-24):** 41 files with
OpenAI C2PA, of which 7 checked by the provenance API were SynthID
`DETECTED` (one `NOT_DETECTED`). Four diverse DETECTED files were staged
under `out/cohort-calibration/openai-expand/` (gitignored; not
publication-cleared) and given a Chroma1 ladder (0.04-0.12, seed 0) by
`chroma_openai_expansion.py` (a harness kept outside this repository):

| id | faces | flat_ratio |
|---|---|---|
| dense_ed68 | 0 | 0.118 |
| midflat_038c | 0 | 0.323 |
| textlike_3b17 | 0 | 0.649 |
| faces2_4dab | 2 | 0.279 |

The provenance API then 429'd. The same checks were finished on
`https://openai.com/research/verify/` (the page reports SynthID and C2PA
separately under View details). Chroma PNG outputs have no C2PA, so a
"No OpenAI signals detected" heading is a pixel-SynthID negative.

| id | faces | @0.04 | @0.05 | @0.06 |
|---|---|---|---|---|
| dense_ed68 | 0 | CLEAN | CLEAN | CLEAN |
| midflat_038c | 0 | -- | -- | CLEAN |
| textlike_3b17 | 0 | -- | -- | CLEAN |
| faces2_4dab | 2 | -- | -- | CLEAN |

9-face grid @ 0.075, seeds 1 and 2: both CLEAN (SynthID not detected).
The committed seed-0 first-clean is not a fluke.

**No OpenAI adaptive arm.** Zero-face extras clear at or below 0.06, the
2-face extra also clears at 0.06, and only the 9-face grid needs 0.075.
That is not a YuNet split: `face_count > 0` would over-strengthen the
2-face file, and a high face-count cutoff would be one fixture. The
spread (0.075 - 0.06) was absorbed by the 0.09 floor for this calibration,
but the later holdout below invalidated that operating point.
Images API generations remain non-carriers (C2PA only).

## OpenAI hard-carrier holdout (2026-09-07)

Two additional withheld OpenAI SynthID carriers expanded the Chroma boundary.
Each original was a positive pixel control after metadata-only stripping, so the
verdict did not depend on C2PA provenance. Their Chroma first-clean strengths were
0.10625 and 0.1375. Both Qwen outputs were clean at the existing 0.07675 operating
point.

The paired fidelity read was mixed rather than a reason to pay Chroma's higher
denoise. On the one detected face, Qwen had higher identity cosine (0.835 vs
0.821), lower face LPIPS (0.108 vs 0.118), and lower whole-image LPIPS (0.095 vs
0.125); Chroma retained more local texture and had slightly higher SSIM. On the
second carrier Qwen again had slightly lower LPIPS (0.104 vs 0.109), while Chroma
had higher SSIM and PSNR.

This invalidates both the 0.09 explicit Chroma floor and the OpenAI auto route.
For explicit Chroma, the face-bearing cohort rule gives
`0.1375 + (0.1375 - 0.075) = 0.20`; that candidate was subsequently verified
clean three times on both holdout carriers. `auto` uses `qwen-zimage` for OpenAI
at its separately calibrated floor; Chroma remains the Microsoft route. The
then-current Qwen floor was 0.07675, superseded by the source-fresh expansion below.

## OpenAI zero-face holdout: the content split does not survive real carriers (2026-09-08)

The `OpenAI content-adaptive expansion` section above closed on fixtures whose
zero-face members first cleared at 0.06 and whose single face member cleared at
0.075, which made faces look like the harder class by a comfortable margin. Three
further withheld zero-face carriers were laddered on the public verifier to test
that, all at ~1.57 MP so image area could not explain a difference, and all
confirmed zero-face by `detect_faces` (YuNet) - the same detector the adaptive
arms gate on, not a downscaled Haar pass, which disagreed with it in both
directions on these very files.

Every regeneration was produced by the deployed worker at seed 0, `mode=all`,
explicit `chroma-zimage`, and every verdict below is the verifier's separate
`SynthID detected` line with `Content Credentials not detected` alongside it, so
no metadata could substitute for a pixel result. Each carrier's own
metadata-stripped, RGB-identical control returned a positive verdict first.

| carrier | detected through | first clean | interval |
|---|---|---|---|
| zero-face A | 0.03 | **0.045** | (0.03, 0.045] |
| zero-face B | 0.045 | **0.06** | (0.045, 0.06] |
| zero-face C | 0.10 | **0.11** | (0.10, 0.11] |

Carrier C was laddered 0.02, 0.03, 0.045, 0.06, 0.075, 0.09, 0.10 detected, then
0.11 and 0.125 clean.

Applying the standard rule to the zero-face class gives
`0.11 + (0.11 - 0.045) = 0.175`; the face class gives
`0.1375 + (0.1375 - 0.075) = 0.20`. **The two classes differ by 0.025 while the
spread within each is 0.065 and 0.0625** - a split would key on a variable that
explains roughly a quarter of the variation it leaves behind. The flat 0.20 floor
therefore stands, and it now covers both content classes rather than only the
face-bearing carriers it was derived from.

The methodological point generalises beyond this cohort. The fixture-only view of
zero-face OpenAI content spanned 0.015; the real carriers span 0.065, four times
wider. A narrow fixture spread produces a small worst-clean-plus-one-spread margin
and a false impression of homogeneity, and it was that impression - not any
measurement - that made a face split look justified here twice.

## Qwen OpenAI source-fresh expansion (2026-09-21)

The same under-sampling defect existed in Qwen's OpenAI floor: two sources first
cleared at 0.06225 and 0.0695, so their 0.00725 spread produced the shipped 0.07675
margin. A later production miss made that n=2 estimate untenable.

Six fresh `gpt-image-2` API generations - three YuNet-positive and three
YuNet-zero-face - all retained SynthID in decoded-RGB-identical metadata-stripped
controls. Their complete Qwen seed-0 ladders expanded the n=8 first-clean range to
0.03125-0.09375, so the standard rule gives
`0.09375 + (0.09375 - 0.03125) = 0.15625`. The face and no-face groups each span
0.05 internally while their worst boundaries differ by only 0.0125, so OpenAI
keeps one flat Qwen floor. The per-source table and exact protocol are recorded in
[`Measured provider boundaries for qwen-zimage`](module-internals.md#measured-provider-boundaries-for-qwen-zimage).

## Content-balanced engine selection check (2026-08-31)

The earlier tables were intentionally carrier-focused and therefore small. A
separate discovery pass tested whether ordinary image content contradicts the
cohort choice at the shipped floors. The tracked matrix contains 19 prompts,
matched across OpenAI `gpt-image-1-mini` and Meta `muse-image-1.0`, spanning
photoreal scenes, faces, action, text-like flat art, products, architecture,
watercolor, and surreal illustration. These files were re-encoded by collection
and are not signal carriers, so this pass measures fidelity only; removal remains
established by the provider-oracle calibration above.

Qwen and Chroma ran in separate sequential H100 invocations, once per source,
with the production provider floor and fixed seed. The comparison isolated the
global stage because both profiles inherit the same source-based Z-Image face
repair. Per-image differences, rather than differences between aggregate means,
were evaluated with exact two-sided sign tests:

| provider | metric | Chroma wins | Qwen wins | two-sided p |
|---|---|---:|---:|---:|
| OpenAI | SSIM | 19 | 0 | 0.0000038 |
| OpenAI | PSNR | 19 | 0 | 0.0000038 |
| OpenAI | MAE | 18 | 1 | 0.000076 |
| OpenAI | edge F1 | 15 | 4 | 0.019 |
| OpenAI | LPIPS | 12 | 7 | 0.359 |
| Meta | LPIPS | 1 | 18 | 0.000076 |
| Meta | edge F1 | 1 | 18 | 0.000076 |
| Meta | SSIM | 5 | 14 | 0.064 |

Meta's remaining pixel metrics disagree: Chroma wins PSNR on 16/19 and MAE on
14/19, while losing both perceptual distance and edge preservation on 18/19 at
its higher removal floor. This is not evidence for a genre classifier. The
direction instead follows the already measured provenance cohort: Chroma for
OpenAI, Qwen for Meta. On OpenAI, the seven LPIPS exceptions include unrelated
classes (landscape, sport, low-light interior, watercolor, monochrome portrait,
surreal art, and brutalism), while Chroma still wins SSIM and PSNR on every one;
there is no coherent input feature that justifies overriding the cohort choice.

Inputs and integrity manifests are under
`data/evaluations/engine-selection/`. Raw generated outputs remain gitignored
under `out/engine-selection-study/`; the `engine_selection_study.py` harness
(kept outside this repository) reproduces the run and
`scripts/analyze_engine_selection_study.py` reproduces
the paired statistics. One generation per prompt is a discovery set, not a
license to fit thresholds. Any future content override must first predict a
rule here and then survive unused generation indices without changing it.

## Oracle session notes

- openai.com/verify throttles anonymous sessions through Cloudflare Turnstile
  after ~15 rapid checks; a fresh browser context clears it. The programmatic
  `POST /v1/content_provenance_checks` API is the better oracle (separate
  SynthID/C2PA outcomes). The `OPENAI_API_KEY` in the main-checkout `.env`
  returned 401 org-wide, including `/v1/models`, through the 2026-08-30
  calibration; it was replaced on 2026-08-31 and both `/v1/models` and
  a provenance check on a committed ChatGPT original then succeeded
  (DETECTED).
- meta.ai/identification enforces a sliding per-IP window: a burst of ~15
  checks exhausted it twice; it reopens minutes later, and the page sometimes
  overstates this as a daily limit. Pacing of roughly 2-3 checks per window
  worked. On 2026-08-31 an anonymous isolated session got a real 429 from
  `POST /api/ai-detector` with page text "You’ve reached the daily limit
  for identifications. Try again tomorrow" after about six checks in the
  UTC morning and again after about fifteen in the UTC afternoon, so the
  daily-limit wording is not always overstated; the afternoon window was
  large enough to finish harvest 2. An evening isolated session the same
  day got ten 200s and finished the leftover harvest-1 rungs plus the
  mosque/botanical seed probes.
- The Gemini app oracle requires the user's logged-in session. A human
  drag-drop of a local PNG works on Work Gemini (u/2, 2026-08-31) and on
  Plus (u/0 the same day); Playwright cannot drive the OS picker, and
  the first `input[type=file].accept` is documents/code only. Plus
  `@synthid` hung once after attach, then returned a tool CLEAN on a
  second human-drag chat. Do not drive the live Chrome while it is in
  active use (wrong-tab risk).

## FLUX.2 klein as a global stage (2026-09-27)

`black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, step-distilled) through
diffusers 0.40.0 `Flux2KleinInpaintPipeline`, all-white mask, the neutral
scrub prompt, guidance 1.0, seed 0, and the Chroma step compensation
(`ceil(4 / strength)` requested, four or five effective steps). Global stage
only, no face repair. Seven tracked originals from
`data/synthid/full-pipeline-quality.csv`; compared with `qwen-zimage` at its
vendor floors (OpenAI 0.15625, Google 0.35) on the same files. Harness:
`flux2_klein_ladder.py`, kept outside this repository.

The inpaint pipeline always appends the clean source latents as in-context
reference tokens (`ref_images = [image_latents_encoded]`), whatever the mask
and strength. With them the model copies the source: PSNR 30-37 dB and LPIPS
0.003-0.02 at every strength up to 0.50, although the start sigma is already
about 0.71 at strength 0.30 on a 1448x1086 image (the resolution-dependent
shift, `mu` 1.48; 0.94 at 2816x1536). The official OpenAI API returned
`detected` on all three OpenAI carriers at 0.30. Stock klein inpainting is
therefore not a remover.

Dropping the reference tokens leaves plain partial-noise img2img. OpenAI
brackets from the official API: one carrier `(0.05, 0.10]`, two
`(0.10, 0.20]`. At 0.20 klein reached PSNR 20.6 dB and LPIPS 0.054-0.058 on
the two limiting carriers, against `qwen-zimage` at 22.4-24.0 dB and
0.037-0.043 at its floor, before klein gets any cross-source margin. For
OpenAI klein is the worse engine. It is fast, under one second per image on a
warm H100.

Google, through Gemini on four accounts (the first returned "a tool error
or quota limit" and later "the usage quota for checks on similar images has
been exceeded", so the rest went to three others, each result recorded
against its own slot): one carrier `(0.05, 0.10]`, one clean at 0.10 with
0.05 unchecked, one `(0.10, 0.20]` ("Part of this image was edited or
generated with Google AI" at 0.10), and one `detected` at 0.10,
"inconclusive" at 0.20 and clean at 0.25. At 0.20 klein matched
`qwen-zimage` at 0.35 on PSNR (17.0-20.3 against 17.6-21.0 dB), beat it on
LPIPS (0.079-0.152 against 0.111-0.191), and kept the objects `qwen-zimage`
redraws (the meatball bowl, the sign lettering). The operating point is what
decides, though: the worst clean boundary 0.25 plus one full observed spread
(0.25 - 0.10) is 0.40, and at 0.40 the limiting carrier reaches LPIPS 0.270
and 14.6 dB against `qwen-zimage`'s 0.183 and 18.9 dB. Even with no margin, at
0.25 it is level (0.185, 16.9 dB). klein is not a replacement global stage
for either vendor.

`scripts/fidelity_metrics.py` (OCR NED against the verified
`data/evaluations/fidelity/ground-truth.json` where it has an entry, OCR of
the original otherwise; ArcFace identity and face-crop sharpness; full-image
LPIPS) at each engine's clean strengths confirms it, and shows where the
512-pixel LPIPS above misled:

| Carrier | Faces | Text | `qwen-zimage` at floor | klein |
| --- | ---: | --- | --- | --- |
| OpenAI face grid | 9 | none | ID 0.853, face sharpness 0.61 | 0.10: ID 0.689; 0.20: ID 0.472 |
| OpenAI typography (2) | 0 | Latin, verified | NED 0.545 / 0.562 | 0.20: NED 0.725 / 0.688 |
| Google party scene | 5 | Latin | ID 0.933, NED 0.386 | 0.10: ID 0.500, NED 0.271; 0.20: ID 0.226, NED 0.450 |
| Google crowd | 18 | none readable | ID 0.752 | 0.10: ID 0.634, face sharpness 0.05 |
| Google sign | 0 | Cyrillic | NED 0.367 | 0.20: NED 0.265 |
| Google CJK | 0 | CJK, verified | NED 0.074 | 0.25: NED 0.296; 0.40: NED 0.667 |

klein keeps Latin and Cyrillic sign lettering better on two Google carriers,
but it loses every face comparison (identity 0.23-0.69 against 0.75-0.93, and
face crops smoothed to 5-18% of the original's Laplacian variance on the
large Google frames) and the verified CJK text. The face gap is partly the
configuration, since `qwen-zimage` includes its source-based face stage and
klein ran global-only; the CJK loss is not. Outputs, reports and
oracle records: `.local-eval/flux2-klein-2026-09-27/`.

## Z-Image-Turbo as a global stage (2026-09-29)

Verdict: **do not replace `qwen-zimage` with Z-Image-Turbo globally**. It is a
promising research direction at the lower OpenAI no-face operating point, but
it is not ready as a general no-face complement: the Google 0.40 outputs have
visible non-face defects. A face detector alone is not a sufficient safety
gate. The narrower route needs a completed Google boundary and a broader
content-fidelity study before it can ship. **Decision: do not integrate it; keep
the harness and measurements as research evidence only.** This decision is
limited to a new global or mflux profile. It does not remove or change the
existing source-conditioned Z-Image face-repair stage shared by
`qwen-zimage`, `sdxl-zimage`, and `chroma-zimage`.

The package already uses `Tongyi-MAI/Z-Image-Turbo` through DiffSynth's
`ZImagePipeline` for source-conditioned face-crop repair after the qwen global
stage. This experiment instead ran the same official Apache-2.0 checkpoint at
revision `f332072aa78be7aecdf3ee76d5c247082da564a6` through diffusers 0.40.0
`ZImageImg2ImgPipeline`. The stock diffusers path supports strength-controlled
img2img and uses the Z-Image scheduler's static shift of 3; unlike klein
inpainting, it does not re-inject clean source latents as reference tokens.
Primary sources: the
[checkpoint](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and
[diffusers pipeline documentation](https://huggingface.co/docs/diffusers/api/pipelines/z_image).

The global-only H100 ladder used seed 0, the neutral scrub prompt, guidance
0.0, eight effective distilled steps, and strengths 0.025, 0.05, 0.10, 0.15,
0.20, 0.30, and 0.40 on the same seven tracked originals as the klein study.
The real shifted start sigmas were 0.0714, 0.1364, 0.25, 0.375, 0.4286, 0.60,
and 0.6667. The 49 generations took 264.5 seconds of inference in total,
about 2.4 seconds per OpenAI image and 7.6 seconds per larger Google image.
The rough Modal H100 cost was $0.29 for inference or $0.40 including the cold
model load. Harness: `z_image_turbo_ladder.py`, kept outside this repository.

The official OpenAI oracle gave first-clean strengths 0.10, 0.10, and 0.15.
The project margin rule therefore selects 0.20, versus the shipped
`qwen-zimage` floor of 0.15625. At those respective operating points,
`scripts/fidelity_metrics.py` found:

| Carrier | `qwen-zimage` | Z-Image-Turbo |
| --- | --- | --- |
| 9-face grid | ID 0.853, face LPIPS 0.054, image LPIPS 0.091 | ID 0.658, face LPIPS 0.092, image LPIPS 0.101 |
| Typography 1 | NED 0.545, image LPIPS 0.123, PSNR 22.5 dB | NED 0.311, image LPIPS 0.086, PSNR 27.3 dB |
| Typography 2 | NED 0.562, image LPIPS 0.116, PSNR 24.0 dB | NED 0.291, image LPIPS 0.074, PSNR 27.7 dB |

Z-Image-Turbo preserves the two no-face typography cards better on every
listed metric, but the face-grid identity loss is materially worse. That is
enough to reject it as the universal OpenAI stage.

Google bracketing is incomplete on one carrier. The party and crowd images
were detected at 0.15 and clean at 0.20. The Cyrillic sign was detected at
0.20 and clean at 0.30. The CJK sign remained detected through 0.15; its 0.20
checks produced two explicit quota failures, and the remaining authenticated
surface could not accept the prepared upload while its activity setting was
off. Those are unreachable results, not clean verdicts. If that carrier first
cleans at 0.20 or 0.30, the margin rule selects 0.40; if it first cleans at
0.40, the rule selects 0.60. Consequently, 0.40 is only a provisional fidelity
probe, not an oracle-established Google operating point.

Against `qwen-zimage` at 0.35, that provisional 0.40 probe shows the same
content split as OpenAI:

| Carrier | Faces | `qwen-zimage` | Z-Image-Turbo |
| --- | ---: | --- | --- |
| Party scene | 5 | ID 0.933, NED 0.386, image LPIPS 0.367 | ID 0.236, NED 0.286, image LPIPS 0.326 |
| Crowd | 18 | ID 0.752, image LPIPS 0.339 | ID 0.448, image LPIPS 0.297 |
| Cyrillic sign | 0 | NED 0.367, LPIPS 0.424, SSIM 0.523 | NED 0.245, LPIPS 0.356, SSIM 0.547 |
| CJK sign | 0 | NED 0.074, LPIPS 0.380, SSIM 0.555 | NED 0.000, LPIPS 0.320, SSIM 0.587 |

Visual inspection agrees with the metrics. At 0.40 Z-Image-Turbo changes age,
facial structure, and skin texture and can introduce doubled facial contours.
The no-face images are not safe either. The CJK output has NED 0.000, but that
only says OCR recovered the same character sequence: the model visibly redraws
the path, foliage, lighting, sign geometry, wood grain, post, and fasteners.
Both engines also make visible Cyrillic errors. The package's existing Z-Image
face stage does not contradict this result. It takes an expanded crop from the
original image, enlarges the detected face toward 768 px, runs DiffSynth at an
adaptive strength of only 0.025-0.14, and composites only the SAM face mask into
the already-clean global result. The global experiment instead ran diffusers at
up to 0.40 over the full frame, where each face was small and every pixel could
be redrawn. The face stage also does not have to remove SynthID: a matched
isolation test found its output alone still detected, while adding it after the
clean Qwen pass improved identity cosine from 0.589 to 0.852. It is a low-noise
masked fidelity repair, not a watermark-removal stage.

As an Apple Silicon side probe, [mflux](https://github.com/mflux-community/mflux)
0.20.0 successfully ran img2img with
`filipstrand/Z-Image-Turbo-mflux-4bit` on an M5 with 32 GB. At 1440x1072,
image strength 0.8 (equivalent denoise fraction 0.20), 40 requested steps and
eight effective steps, model load took 5.3 seconds and two deterministic runs
took 301.5 and 380.0 seconds. The pixel outputs were identical, but there was
no warm-run speedup. This is slow for an interactive default, but it is also the
only measured native Apple Silicon image-regeneration path: every shipped
invisible-image profile is CUDA-only and refuses MPS or CPU. The mflux route
therefore proves M5 feasibility, but the visible defects at the tested Google
strength keep it from being a usable fallback. Shipping it would require a
lower safe operating point, the missing Google boundary, broader content
fidelity, packaging, and product-level tests. Outputs, fidelity logs,
manifests, and the local timing report are under
`.local-eval/z-image-turbo-2026-09-27/`.

## Qwen-Image-2.1 as a global stage (2026-10-29)

Verdict: **ship `qwen21-zimage` (`--pipeline qwen21-zimage`) as an explicit
opt-in two-stage profile for non-commercial research, evaluation, and personal
use only; keep `qwen-zimage` (`Qwen/Qwen-Image-2512`, Apache-2.0) as the default
profile and keep `--pipeline auto` restricted to Apache-2.0 / commercial-safe
profiles**. Although Hugging Face `diffusers` exposes `QwenImageEditPipeline`
with `image=` only as reference conditioning starting from pure noise,
DiffSynth-Studio (`diffsynth==2.1.9`, the framework already used by
`qwen-zimage`) supports native partial-noise `input_image` +
`denoising_strength` (`QwenImageUnit_Image2Image`), structural Canny ControlNet
conditioning (`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`, Apache-2.0),
the community texture-repaired decoder
(`madebyollin/texture-fix-vae-for-qwen-image-2.1`, Apache-2.0), and two-phase
`edit_image` reference switching (issue #111). Pairing the four-step unfused
`Qwen-Image-2.1` + `texture-fix-vae` + Canny global pass with the shared
`Z-Image-Turbo` face stage (`Qwen21ZImagePipeline`) beats `qwen-zimage` across
all four provider cohorts (OpenAI SynthID, Google SynthID, Microsoft InvisMark,
and Meta Content Seal), beats or ties `sdxl-zimage-0.50` across all four Google
SynthID carriers at `s=0.35`, and beats `chroma-zimage-0.125` on the
face-bearing Microsoft Designer carrier (`ID cos = 0.915` vs `0.889`, `img LPIPS = 0.224`
vs `0.244`). However, because `Qwen/Qwen-Image-2.1`, `Qwen/Qwen-Image-2.1-Turbo`,
and `lightx2v/Qwen-Image-2.1-Lightning` are governed by the **Qwen Research
License** (non-commercial research and evaluation only without a separate
commercial agreement from Alibaba Cloud), `qwen21-zimage` is never selected by
`--pipeline auto`.

### Setup and DiffSynth 2.1.9 LoRA loader trap

All runs executed on Modal H100 in bf16 at native geometry (dimensions floored
to the `/16` latent grid and restored if needed), seed 0, CFG 1.0, and the
neutral scrub prompt (`high quality, sharp, detailed, faithful to the original`).
Harness, outputs, manifests, and hash-bound oracle records live under
`.local-eval/qwen-image-21-2026-10-29/` (`qwen21_ladder.py`).

Three distinct modes were evaluated across the seven tracked originals (plus
visible-mark-cleaned Google controls for `gemini-web` SynthID verification):

1. **`vae-roundtrip` (`vae-stock` vs `vae-tfix`):** Pure 64-channel 16x RGBA
   VAE encode-decode with zero diffusion steps, comparing `Qwen/Qwen-Image-2.1`
   (`vae/diffusion_pytorch_model.safetensors`, `z_dim=64`, spatial `16x`)
   against `madebyollin/texture-fix-vae-for-qwen-image-2.1` (`alpha_tfix=1.0`
   decoder weights).
2. **`canny-lightning` (`canny-l4` / `canny-base4` vs `canny-fused-l4`, each
   with stock and `tfix` VAEs):** Four effective steps on
   `FlowMatchScheduler("Qwen-Image-Lightning")` with
   `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` (`scale=1.0` on OpenCV
   Canny `100/200`), comparing the unfused base `Qwen/Qwen-Image-2.1`
   transformer against the transformer with all 128 fused
   `lightx2v/Qwen-Image-2.1-Lightning` (`Qwen-Image-2.1-Lightning-4steps-V1.0-bf16.safetensors`,
   rank 64, `alpha=0.8`) attention projections (`1.53-1.84 s/image` on H100).
3. **`ref-switch` (`refswitch-base40` and `refswitch-l8-canny`):** Two-phase
   mid-trajectory reference switching reproducing the ComfyUI dual-`KSampler`
   workflow proposed by `@dichiara19` in issue #111. Phase 1 runs with
   `edit_image=source` concatenated into the DiT token sequence while
   $\sigma > \sigma_{\text{switch}}$; Phase 2 drops `edit_latents` (`None`) and
   continues unreferenced from the Phase 1 latent state for
   $\sigma \le \sigma_{\text{switch}}$ (`13.4 s/image` at `s=0.40` on 40-step
   base `Qwen-Image-2.1`).

**DiffSynth 2.1.9 LoRA trap:** `GeneralLoRALoader.get_name_dict` in
`diffsynth/utils/lora/general.py` checks `if ".lora_up." in key:` (underscore)
and strips `diffusion_model.`, whereas
`lightx2v/Qwen-Image-2.1-Lightning` stores keys as
`transformer.transformer_blocks.<idx>.attn.<proj>.lora.up.weight` and
`.lora.down.weight` (dot). Calling `pipe.load_lora(...)` silently fuses **0
tensors** and runs the unfused base transformer. Stripping `transformer.` and
matching `.lora.up.weight` / `.lora.down.weight` explicitly fused all
**`128/128`** linear projections (`to_q`, `to_k`, `to_v`, `to_out.0` across 32
blocks). Both the unfused (`canny-l4` / `canny-base4`) and 128-layer fused
(`canny-fused-l4`) runs were retained and compared head-to-head.

### 1. 64-channel 16x RGBA VAE ceiling (`vae-roundtrip`)

Across all seven originals, `Qwen/Qwen-Image-2.1`'s 64-channel 16x RGBA VAE
reaches a mean PSNR of **35.68 dB** (`32.27-37.31 dB`), mean SSIM **0.9201**,
and mean 512-pixel LPIPS **0.0010** -- +6.27 dB higher PSNR and 25x lower
512-pixel LPIPS than `FLUX.2-klein`'s 32-channel VAE (`29.41 dB`, `0.0258`).
On `scripts/fidelity_metrics.py`, `madebyollin/texture-fix-vae-for-qwen-image-2.1`
(`vae-tfix`) substantially improves full-resolution perceptual distance, face
texture, and OCR over the stock VAE:

| Carrier | Variant | text NED | ID cos | face LPIPS | lapvar | img LPIPS | SSIM | PSNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI light poster (`02_02_23 PM`) | `vae-stock` | 0.092 | - | - | - | 0.030 | 0.903 | 36.8 |
| OpenAI light poster (`02_02_23 PM`) | `vae-tfix` | **0.079** | - | - | - | **0.017** | **0.916** | 36.7 |
| Google 18-face grid (`y48j3c`) | `vae-stock` | - | 0.961 | 0.039 | 0.728 | 0.061 | 0.870 | **35.3** |
| Google 18-face grid (`y48j3c`) | `vae-tfix` | - | **0.967** | **0.018** | **0.831** | **0.025** | 0.869 | 34.4 |

Pure VAE encode-decode does not remove SynthID (`vae-stock` on `02_02_23 PM`
returned `detected` on the official OpenAI provenance API), but the 64-channel
bottleneck plus `vae-tfix` removes the VAE reconstruction floor as a source of
glyph and skin-texture loss.

### 2. Provider-oracle boundaries (OpenAI API and Google `@SynthID`)

OpenAI checks used `scripts/provider_oracles.py check-openai` (`POST /v1/content_provenance_checks`,
SynthID read separately from C2PA). Google checks used `scripts/provider_oracles.py prepare` /
`record` / `verify` on `gemini-web` across three authenticated browser slots
(`personal-u0`, `veedma-u1`, `lorsey-u2`) against visible-mark-cleaned,
metadata-stripped controls (`3mc4t9` positive control verified `detected` in
all three accounts before testing candidates; `synthid.com` returned `Please
sign in before detection.` in both anonymous and signed-in Chrome sessions):

| Cohort | Carrier(s) | Mode & Variant | `s=0.05` | `s=0.10` | `s=0.15625` | `s=0.20` | `s=0.25` | `s=0.35` | `s=0.40` | `s=0.45` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI | all 3 (`10_31_08`, `02_02_23`, `02_03_55`) | `canny-l4` (no LoRA) | 3/3 `detected` | **3/3 `not_detected`** | 3/3 `not_detected` | 3/3 `not_detected` | 3/3 `not_detected` | - | - | - |
| OpenAI | all 3 (`10_31_08`, `02_02_23`, `02_03_55`) | `canny-l4-tfix` (no LoRA + tfix VAE) | - | **3/3 `not_detected`** | - | - | - | - | - | - |
| OpenAI | all 3 (`10_31_08`, `02_02_23`, `02_03_55`) | `canny-fused-l4` (128/128 LoRA) | 3/3 `detected` | **3/3 `not_detected`** | 3/3 `not_detected` | - | - | - | - | - |
| OpenAI | all 3 (`10_31_08`, `02_02_23`, `02_03_55`) | `refswitch-base40` (`sw=0.30` & `sw=0.40`) | - | - | - | - | **3/3 `not_detected`** | - | - | - |
| Google | `3mc4t9` (5-face party scene) | `canny-fused-l4` (128/128 LoRA) | - | **`not_detected`** (`u/2`) | - | **`not_detected`** (`u/0`) | - | **`not_detected`** (`u/0`) | - | **`not_detected`** (`u/0`) |
| Google | `3mc4t9` (5-face party scene) | `canny-base4` (no LoRA) | - | - | - | - | - | **`not_detected`** (`u/1`) | - | - |
| Google | `3mc4t9` (5-face party scene) | `refswitch-base40-sw0.30` (`ref=3, noref=7`) | - | - | - | - | `detected` ("Part of...", `u/2`) | - | - | - |
| Google | `3mc4t9` (5-face party scene) | `refswitch-base40-sw0.40` & `sw0.46` (`ref=7/5, noref=9/11`) | - | - | - | - | - | - | **`not_detected`** (`u/1`) | - |

Two mechanism findings stand out:

- **Four-step Canny lowers the OpenAI first-clean boundary across all three
  tracked carriers to `(0.05, 0.10]`.** Unlike `Z-Image-Turbo` (where the
  9-face grid `10_31_08 AM` stayed `detected` through `0.15` and only cleared
  at `0.20`), both `canny-l4` and `canny-fused-l4` clear all three OpenAI
  originals at `s=0.10` ($\sigma_0 = 0.2500$).
- **Two-phase reference switching (`refswitch-base40`) confirms both the
  mechanism and the Google threshold reported in issue #111.** Keeping
  `edit_image` active during the high-noise prefix anchors global structure so
  strongly that `s=0.40, sw=0.30` (`ref=9, noref=7`) produces nearly the same
  PSNR (`27.67 dB` mean) as `s=0.25, sw=0.30` (`ref=3, noref=7`, `27.78 dB`
  mean). However, because `edit_latents` are the watermarked source latents,
  keeping them down to $\sigma_{\text{switch}} = 0.30$ (`7` unreferenced tail
  steps) only clears OpenAI and leaves Google SynthID partially `detected` on
  `3mc4t9`; raising the handoff to `s=0.40, sw=0.40` (`9` unreferenced steps)
  or `sw=0.46` (`11` unreferenced steps) is required to reach `not_detected` on
  Google.

### 3. Fidelity comparison (`scripts/fidelity_metrics.py`)

At their respective oracle-clean OpenAI rungs (`s=0.10` for four-step Canny,
`s=0.25, sw=0.30` for 40-step reference switching), compared against shipped
`qwen-zimage` (`0.15625`), `sdxl-zimage` (`0.15`), and `Z-Image-Turbo` (`0.20`):

| Carrier | Variant | Oracle | text NED | ID cos | face LPIPS | lapvar | img LPIPS | SSIM | PSNR (dB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI light poster (`02_02_23 PM`, OCR floor `0.127`) | `canny-l4` `s=0.10` (no LoRA) | `not_detected` | **0.136** | - | - | - | 0.057 | 0.859 | **28.3** |
| OpenAI light poster (`02_02_23 PM`, OCR floor `0.127`) | `canny-l4-tfix` `s=0.10` | `not_detected` | 0.178 | - | - | - | **0.053** | **0.865** | **28.3** |
| OpenAI light poster (`02_02_23 PM`, OCR floor `0.127`) | `canny-fused-l4-tfix` `s=0.10` | `not_detected` | 0.258 | - | - | - | 0.063 | 0.837 | 27.6 |
| OpenAI light poster (`02_02_23 PM`, OCR floor `0.127`) | `refswitch-base40` `s=0.25, sw=0.30` | `not_detected` | 0.273 | - | - | - | **0.051** | 0.847 | 27.7 |
| OpenAI dark poster (`02_03_55 PM`, OCR floor `0.214`) | `canny-l4-tfix` `s=0.10` | `not_detected` | **0.299** | - | - | - | **0.052** | **0.861** | **28.5** |
| OpenAI dark poster (`02_03_55 PM`, OCR floor `0.214`) | `canny-l4` `s=0.10` (no LoRA) | `not_detected` | 0.336 | - | - | - | 0.055 | 0.855 | **28.5** |
| OpenAI dark poster (`02_03_55 PM`, OCR floor `0.214`) | `canny-fused-l4-tfix` `s=0.10` | `not_detected` | 0.328 | - | - | - | 0.061 | 0.840 | 27.9 |
| OpenAI dark poster (`02_03_55 PM`, OCR floor `0.214`) | `refswitch-base40` `s=0.25, sw=0.30` | `not_detected` | 0.326 | - | - | - | **0.052** | 0.841 | 27.7 |
| OpenAI 9-face grid (`10_31_08 AM`, global only) | `canny-l4` `s=0.10` (no LoRA) | `not_detected` | - | **0.803** | **0.058** | 0.969 | **0.057** | 0.872 | **31.4** |
| OpenAI 9-face grid (`10_31_08 AM`, global only) | `canny-fused-l4` `s=0.10` | `not_detected` | - | 0.798 | 0.062 | **0.992** | 0.062 | 0.860 | 30.7 |
| OpenAI 9-face grid (`10_31_08 AM`, global only) | `refswitch-base40` `s=0.25, sw=0.30` | `not_detected` | - | 0.772 | 0.060 | 0.981 | 0.064 | 0.844 | 30.1 |

At `s=0.10` ($\sigma_0 = 0.25$), four Euler steps already have a small average
step size ($\Delta\sigma = 0.0625$, equivalent to a 16-step full trajectory),
so the unfused base `Qwen-Image-2.1` transformer + Canny (`canny-l4` /
`canny-l4-tfix`) preserves small poster typography better (`0.136` and `0.299`
NED) than the 128-layer fused 4-step Lightning LoRA (`0.258` and `0.328` NED),
whose velocity amplification is tuned for full $\sigma_0 = 1.0 \to 0$
generation.

On the Google face carriers (`3mc4t9` 5-face party scene and `y48j3c` 18-face
portrait grid, global-only):

| Carrier | Variant | Oracle (`3mc4t9`) | faces | ID cos | face LPIPS | lapvar | img LPIPS | SSIM | PSNR (dB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Google 5-face (`3mc4t9`) | `canny-fused-l4` `s=0.10` | `not_detected` | 5/5 | **0.723** | **0.066** | 0.729 | **0.098** | **0.733** | **23.5** |
| Google 5-face (`3mc4t9`) | `canny-fused-l4` `s=0.20` | `not_detected` | 5/5 | 0.573 | 0.085 | 0.722 | 0.142 | 0.665 | 21.5 |
| Google 5-face (`3mc4t9`) | `canny-fused-l4` `s=0.35` | `not_detected` | 5/5 | 0.443 | 0.105 | 0.678 | 0.188 | 0.616 | 19.9 |
| Google 5-face (`3mc4t9`) | `canny-base4` `s=0.35` (no LoRA) | `not_detected` | 5/5 | 0.535 | 0.104 | 0.551 | 0.178 | 0.656 | 21.1 |
| Google 5-face (`3mc4t9`) | `refswitch-base40` `s=0.40, sw=0.40` | `not_detected` | 5/5 | 0.646 | 0.070 | **0.750** | 0.117 | 0.665 | 21.9 |
| Google 5-face (`3mc4t9`) | `refswitch-base40` `s=0.40, sw=0.46` | `not_detected` | 5/5 | 0.588 | 0.078 | 0.749 | 0.144 | 0.634 | 21.0 |
| Google 18-face (`y48j3c`) | `canny-fused-l4-tfix` `s=0.10` | - | 18/18 | **0.773** | **0.062** | 0.804 | **0.099** | **0.710** | 26.7 |
| Google 18-face (`y48j3c`) | `canny-fused-l4-tfix` `s=0.20` | `indeterminate` | 18/18 | 0.691 | 0.083 | 0.747 | 0.137 | 0.656 | 25.1 |
| Google 18-face (`y48j3c`) | `canny-fused-l4-tfix` `s=0.35` | - | 18/18 | 0.610 | 0.096 | 0.685 | 0.164 | 0.621 | 23.8 |
| Google 18-face (`y48j3c`) | `refswitch-base40` `s=0.25, sw=0.30` | - | 18/18 | **0.773** | 0.066 | 0.814 | 0.105 | 0.703 | **27.1** |
| Google 18-face (`y48j3c`) | `refswitch-base40` `s=0.40, sw=0.40` | - | 18/18 | 0.706 | 0.078 | **0.815** | 0.128 | 0.666 | 25.8 |
| Google 18-face (`y48j3c`) | `refswitch-base40` `s=0.40, sw=0.46` | - | 18/18 | 0.651 | 0.087 | 0.814 | 0.145 | 0.644 | 24.9 |

Even with the 64-channel RGBA VAE and `madebyollin` texture-fix decoder, a
global-only `Qwen-Image-2.1` pass drops ArcFace identity cosine to `0.588-0.723`
on `3mc4t9` and `0.610-0.773` on `y48j3c`, versus `0.933` and `0.852` for
two-stage `qwen-zimage` (whose source-conditioned `Z-Image-Turbo` face stage
restores masked face crops after the global scrub). At the higher Google
strengths (`s=0.35-0.40`), `refswitch-base40` (`sw=0.40`) retains notably higher
face identity (`0.646` vs `0.443` on `3mc4t9`, `0.706` vs `0.610` on `y48j3c`)
and lower whole-image LPIPS (`0.117` vs `0.188`) than four-step Canny at
`s=0.35`, at the cost of 7.4x slower H100 inference (`13.4 s` vs `1.8 s`).

#### Two-stage `qwen21-zimage` (`Qwen-Image-2.1` + `texture-fix-vae` + Canny + `Z-Image-Turbo` face stage)

Pairing the four-step `Qwen-Image-2.1` + `madebyollin/texture-fix-vae-for-qwen-image-2.1` + `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` Canny global stage with the shared `Z-Image-Turbo` + SAM + YuNet face stage (`--pipeline qwen21-zimage`) restores face identity while keeping `Qwen-Image-2.1`'s stronger background fidelity:

| Carrier | Two-stage variant | Strength | Oracle | faces | ID cos | face LPIPS | lapvar | img LPIPS | SSIM | PSNR (dB) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Google 18-face (`y48j3c`) | `qwen21-zimage` (unfused) | `0.20` | - | 18/18 | **0.890** | **0.038** | 0.536 | **0.125** | **0.710** | **26.4** |
| Google 18-face (`y48j3c`) | `qwen21-zimage` (unfused) | `0.35` | - | 18/18 | **0.874** | **0.043** | 0.515 | **0.175** | **0.684** | **24.9** |
| Google 18-face (`y48j3c`) | `qwen21-zimage-lora` (fused) | `0.20` | - | 18/18 | 0.877 | 0.042 | 0.659 | 0.140 | 0.678 | 25.4 |
| Google 18-face (`y48j3c`) | `qwen21-zimage-lora` (fused) | `0.35` | - | 18/18 | 0.851 | 0.050 | 0.616 | 0.193 | 0.648 | 24.0 |
| Google 18-face (`y48j3c`) | `qwen-zimage` (`Qwen-Image-2512`) | `0.20` | `not_detected` | 18/18 | 0.877 | 0.040 | 0.525 | 0.252 | 0.659 | 24.8 |
| OpenAI 9-face grid (`10_31_08 AM`) | `qwen21-zimage` (unfused) | `0.15625` | `detected` (`req_3b58ff787cee4d22ba2934cb7e7e8ae4`) | 9/9 | **0.863** | **0.051** | 0.630 | **0.071** | **0.875** | **31.3** |
| OpenAI 9-face grid (`10_31_08 AM`) | `qwen21-zimage` (unfused) | `0.20` | **`not_detected`** (`req_2e599929c32747a5b1a5a6c39623b802`) | 9/9 | 0.840 | 0.055 | 0.625 | 0.081 | 0.867 | 30.8 |
| OpenAI 9-face grid (`10_31_08 AM`) | `qwen21-zimage` (unfused) | `0.25` | `not_detected` (`req_5712ed8cb38e42ae9ba60df7c915659f`) | 9/9 | 0.819 | 0.060 | 0.620 | 0.091 | 0.858 | 30.3 |
| OpenAI 9-face grid (`10_31_08 AM`) | `qwen-zimage` (`Qwen-Image-2512`) | `0.15625` | `not_detected` | 9/9 | 0.842 | 0.064 | 0.562 | 0.145 | 0.805 | 28.0 |

Head-to-head across all four provider cohorts at each profile's certified operating floor (`qwen21-zimage`: `0.20` OpenAI / `0.35` Google / `0.15` Microsoft / `0.10` Meta; `qwen-zimage`: `0.15625` / `0.35` / `0.15` / `0.10`; `sdxl-zimage`: `0.15` / `0.50` / `0.15`; `chroma-zimage`: `0.20` / `0.50` / `0.125` / `0.17`):

| Cohort & Carrier | Profile & Floor | Oracle | text NED | faces | ID cos | face LPIPS | lapvar | img LPIPS | SSIM | PSNR (dB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Google 5-face (`3mc4t9`) | `qwen21-zimage` (`0.35`) | `not_detected` | - | 5/5 | **0.961** | 0.024 | 0.621 | **0.194** | 0.648 | 20.7 |
| Google 5-face (`3mc4t9`) | `sdxl-zimage` (`0.50`) | `not_detected` | - | 5/5 | **0.961** | **0.021** | 0.630 | 0.201 | **0.659** | **21.3** |
| Google 5-face (`3mc4t9`) | `qwen-zimage` (`0.35`) | `not_detected` | - | 5/5 | 0.932 | 0.031 | 0.600 | 0.373 | 0.526 | 18.1 |
| Google 5-face (`3mc4t9`) | `chroma-zimage` (`0.50`) | `not_detected` | - | 5/5 | 0.911 | 0.051 | **0.656** | 0.423 | 0.477 | 18.2 |
| Google 18-face (`y48j3c`) | `qwen21-zimage` (`0.35`) | `not_detected` | - | 18/18 | **0.874** | **0.043** | 0.515 | **0.175** | **0.684** | **24.9** |
| Google 18-face (`y48j3c`) | `sdxl-zimage` (`0.50`) | `not_detected` | - | 18/18 | 0.847 | **0.043** | 0.496 | 0.186 | 0.670 | 24.2 |
| Google 18-face (`y48j3c`) | `qwen-zimage` (`0.35`) | `not_detected` | - | 18/18 | 0.757 | 0.065 | 0.486 | 0.335 | 0.579 | 21.1 |
| Google 18-face (`y48j3c`) | `chroma-zimage` (`0.50`) | `not_detected` | - | 18/18 | 0.788 | 0.062 | **0.533** | 0.340 | 0.568 | 22.4 |
| Google CJK sign (`633uuy`) | `qwen21-zimage` (`0.35`) | `not_detected` | **0.038** | 0 | - | - | - | 0.208 | 0.733 | 24.1 |
| Google CJK sign (`633uuy`) | `sdxl-zimage` (`0.50`) | `not_detected` | **0.038** | 0 | - | - | - | **0.190** | **0.744** | **24.3** |
| Google CJK sign (`633uuy`) | `qwen-zimage` (`0.35`) | `not_detected` | 0.115 | 0 | - | - | - | 0.359 | 0.564 | 19.3 |
| Google CJK sign (`633uuy`) | `chroma-zimage` (`0.50`) | `not_detected` | 0.074 | 0 | - | - | - | 0.404 | 0.582 | 20.7 |
| Google Cyrillic sign (`akdbei`) | `qwen21-zimage` (`0.35`) | `not_detected` | - | 0 | - | - | - | **0.196** | **0.713** | **24.6** |
| Google Cyrillic sign (`akdbei`) | `sdxl-zimage` (`0.50`) | `not_detected` | - | 0 | - | - | - | 0.210 | 0.710 | 24.0 |
| Google Cyrillic sign (`akdbei`) | `qwen-zimage` (`0.35`) | `not_detected` | - | 0 | - | - | - | 0.408 | 0.529 | 18.8 |
| Google Cyrillic sign (`akdbei`) | `chroma-zimage` (`0.50`) | `not_detected` | - | 0 | - | - | - | 0.438 | 0.547 | 20.8 |
| Microsoft InvisMark (`ms-invismark`) | `qwen21-zimage` (`0.125` / `0.15`) | `detected` / **`not_detected`** | - | 0 | - | - | - | **0.086** / 0.100 | 0.879 / 0.870 | 29.3 / 28.6 |
| Microsoft InvisMark (`ms-invismark`) | `chroma-zimage` (`0.125`) | `not_detected` | - | 0 | - | - | - | 0.088 | **0.895** | **30.6** |
| Microsoft InvisMark (`ms-invismark`) | `qwen-zimage` (`0.15`) | `not_detected` | - | 0 | - | - | - | 0.127 | 0.816 | 25.7 |
| Microsoft InvisMark (`ms-invismark`) | `sdxl-zimage` (`0.15`) | `not_detected` | - | 0 | - | - | - | 0.143 | 0.847 | 28.0 |
| Microsoft Paint (`ms-paint-invismark`) | `qwen21-zimage` (`0.10` / `0.15`) | **`not_detected`** / **`not_detected`** | - | 0 | - | - | - | **0.003** / **0.003** | **0.996** / **0.996** | 46.9 / 45.7 |
| Microsoft Paint (`ms-paint-invismark`) | `chroma-zimage` (`0.125`) | `not_detected` | - | 0 | - | - | - | 0.004 | **0.996** | **47.7** |
| Microsoft Paint (`ms-paint-invismark`) | `qwen-zimage` (`0.15`) | `not_detected` | - | 0 | - | - | - | 0.008 | 0.994 | 35.8 |
| Microsoft Paint (`ms-paint-invismark`) | `sdxl-zimage` (`0.15`) | `not_detected` | - | 0 | - | - | - | 0.005 | **0.996** | 43.0 |
| Microsoft Bing (`ms-bing-creator`) | `qwen21-zimage` (`0.08` / `0.15`) | **`not_detected`** / **`not_detected`** | - | 0 | - | - | - | **0.073** / 0.113 | **0.845** / 0.810 | **30.1** / 29.1 |
| Microsoft Bing (`ms-bing-creator`) | `chroma-zimage` (`0.125`) | `not_detected` | - | 0 | - | - | - | 0.094 | 0.834 | 29.6 |
| Microsoft Bing (`ms-bing-creator`) | `qwen-zimage` (`0.15`) | `not_detected` | - | 0 | - | - | - | 0.165 | 0.760 | 27.3 |
| Microsoft Bing (`ms-bing-creator`) | `sdxl-zimage` (`0.15`) | `not_detected` | - | 0 | - | - | - | 0.158 | 0.779 | 28.6 |
| Microsoft Designer (`ms-designer-1a206423`) | `qwen21-zimage` (`0.125` / `0.15`) | **`not_detected`** / **`not_detected`** | - | 1/1 | 0.910 / **0.915** | **0.024** / 0.026 | **0.610** / 0.600 | **0.202** / 0.224 | **0.582** / 0.549 | **21.6** / 21.2 |
| Microsoft Designer (`ms-designer-1a206423`) | `chroma-zimage` (`0.125`) | `not_detected` | - | 1/1 | 0.889 | 0.030 | 0.562 | 0.244 | 0.551 | 21.1 |
| Microsoft Designer (`ms-designer-1a206423`) | `qwen-zimage` (`0.15`) | `not_detected` | - | 1/1 | 0.885 | 0.031 | 0.551 | 0.258 | 0.441 | 19.6 |
| Microsoft Designer (`ms-designer-1a206423`) | `sdxl-zimage` (`0.15`) | `not_detected` | - | 1/1 | 0.909 | 0.026 | 0.576 | 0.233 | 0.531 | 21.2 |
| Meta Content Seal (`meta-lighthouse`) | `qwen21-zimage` (`0.06` / `0.10`) | `detected` / `refused` (quota) | - | 0 | - | - | - | 0.067 / **0.107** | 0.920 / **0.884** | 32.0 / **29.6** |
| Meta Content Seal (`meta-lighthouse`) | `qwen-zimage` (`0.10`) | `not_detected` | - | 0 | - | - | - | 0.172 | 0.785 | 25.7 |
| Meta Content Seal (`meta-lighthouse`) | `chroma-zimage` (`0.17`) | `not_detected` | - | 0 | - | - | - | 0.248 | 0.781 | 26.3 |
| Meta Content Seal (`meta-fox`) | `qwen21-zimage` (`0.10`) | `refused` (quota) | - | 0 | - | - | - | **0.101** | **0.876** | **29.0** |
| Meta Content Seal (`meta-fox`) | `qwen-zimage` (`0.10`) | `not_detected` | - | 0 | - | - | - | 0.166 | 0.777 | 25.7 |
| Meta Content Seal (`meta-fox`) | `chroma-zimage` (`0.17`) | `not_detected` | - | 0 | - | - | - | 0.214 | 0.770 | 26.0 |
| Meta Content Seal (`meta-night-city`) | `qwen21-zimage` (`0.10`) | `refused` (quota) | - | 0 | - | - | - | **0.128** | **0.851** | **27.0** |
| Meta Content Seal (`meta-night-city`) | `qwen-zimage` (`0.10`) | `not_detected` | - | 0 | - | - | - | 0.190 | 0.748 | 23.9 |
| Meta Content Seal (`meta-night-city`) | `chroma-zimage` (`0.17`) | `not_detected` | - | 0 | - | - | - | 0.268 | 0.721 | 23.7 |
| Meta Content Seal (`meta-studio-mug`) | `qwen21-zimage` (`0.10`) | **`not_detected`** | - | 0 | - | - | - | **0.016** | **0.990** | **46.0** |
| Meta Content Seal (`meta-studio-mug`) | `qwen-zimage` (`0.10`) | `not_detected` | - | 0 | - | - | - | 0.024 | 0.984 | 39.3 |
| Meta Content Seal (`meta-studio-mug`) | `chroma-zimage` (`0.17`) | `not_detected` | - | 0 | - | - | - | 0.022 | 0.987 | 36.6 |
| Meta Content Seal (`meta-text-poster`) | `qwen21-zimage` (`0.10`) | **`not_detected`** | 0.014 | 0 | - | - | - | **0.006** | **0.990** | **38.4** |
| Meta Content Seal (`meta-text-poster`) | `qwen-zimage` (`0.10`) | `not_detected` | **0.007** | 0 | - | - | - | 0.011 | 0.984 | 34.0 |
| Meta Content Seal (`meta-text-poster`) | `chroma-zimage` (`0.17`) | `not_detected` | **0.007** | 0 | - | - | - | 0.008 | 0.988 | 36.8 |

Four findings govern the shipped `qwen21-zimage` configuration:

1. **Unfused 4-step Canny beats fused `Qwen-Image-2.1-Lightning` LoRA in two-stage mode as well as single-stage.** On `y48j3c` at `s=0.35`, unfused `qwen21-zimage` scores higher ArcFace cosine (`0.874` vs `0.851`), lower face LPIPS (`0.043` vs `0.050`), lower whole-image LPIPS (`0.175` vs `0.193`), higher SSIM (`0.684` vs `0.648`), and `+0.9 dB` higher PSNR (`24.9` vs `24.0 dB`), while avoiding the 1.48 GB `Qwen-Image-2.1-Lightning` LoRA download and 128-layer fuse step.
2. **OpenAI two-stage floor is `0.20` (`QWEN21_ZIMAGE_OPENAI_STRENGTH = 0.20`).** On `10_31_08 AM` (where 9 `Z-Image-Turbo` face boxes cover 46.2% of the frame), the global pass alone cleared at `s=0.10` and `s=0.15625` (`req_e6a46a510ac94a66a2ac425f51e5d678`), and the zero-face OpenAI posters (`02_02_23 PM`, `02_03_55 PM`) cleared at both `0.10` and `0.15625` (`req_7bca9f38380e4a7fa9d15ed3e0c0f970`, `req_996cc315a4864ed9bfe4bb5bd653f9ed`). However, because `Qwen-Image-2.1` + `texture-fix-vae` preserves ~3.9 dB more background signal at `s=0.15625` (`31.7 dB` vs `27.8 dB` for `Qwen-Image-2512`), compositing 9 source-cropped `Z-Image-Turbo` face patches (`face_strength=0.1144`) over the `s=0.15625` background left the composite `detected` (`req_3b58ff787cee4d22ba2934cb7e7e8ae4`). Raising the global OpenAI floor to `0.20` cleared the two-stage composite (`not_detected`, `req_2e599929c32747a5b1a5a6c39623b802`) while still retaining `30.8 dB` PSNR and `0.081` image LPIPS (`+2.8 dB` and `-0.064` LPIPS better than `qwen-zimage` at `0.15625`).
3. **Google (`0.35`), Microsoft (`0.15`), and Meta (`0.10`) floors match `qwen-zimage` while delivering higher fidelity across all four cohorts.** On Google SynthID, `qwen21-zimage` at `0.35` beats or ties `sdxl-zimage` at `0.50` across all four tracked carriers (`ID cos = 0.874` vs `0.847` and `PSNR = 24.9` vs `24.2 dB` on 18-face `y48j3c`; `ID cos = 0.961` tied and `img LPIPS = 0.194` vs `0.201` on 5-face `3mc4t9`; `text NED = 0.038` tied on CJK `633uuy`; `img LPIPS = 0.196` vs `0.210` and `PSNR = 24.6` vs `24.0 dB` on Cyrillic `akdbei`). On Microsoft InvisMark (`microsoft-web` at `ai.azure.com/nextgen/validate`), `qwen21-zimage` clears `ms-designer-1a206423` at `0.06`, `ms-bing-creator` at `0.08`, `ms-paint-invismark` at `0.10`, and `ms-invismark` at `0.15` (4/4 `not_detected` at `0.15`), beating `chroma-zimage-0.125` on the face-bearing Designer carrier (`ID cos = 0.915` vs `0.889`, `img LPIPS = 0.224` vs `0.244`) while trailing `chroma-zimage-0.125` slightly on the zero-face Paint/Bing/InvisMark carriers (`+0.5` to `+2.0 dB` PSNR in Chroma's favor due to its lower `0.125` floor). On Meta Content Seal, `meta-lighthouse` remained `detected` at `0.06`, while at `0.10` `qwen21-zimage` beats `qwen-zimage-0.10` and `chroma-zimage-0.17` across all five carriers by `+3.1` to `+6.7 dB` PSNR (`27.0-46.0 dB` vs `23.9-39.3 dB`).
4. **`madebyollin/texture-fix-vae-for-qwen-image-2.1` requires `bfloat16` storage (`_build_qwen21_vram_config`), never `float8_e4m3fn` disk/CPU offload.** To prevent fp16 activation overflow, `texture-fix-vae` rescales its decoder weights by `1 / 512` (`2^-9`) and encoder weights by `1 / 8`. Passing `offload_dtype=torch.float8_e4m3fn` causes DiffSynth's `DiskMap` to cast every VAE weight tensor to unscaled `float8_e4m3fn` at load time, underflowing the `1 / 512`-scaled decoder weights below the minimum `e4m3fn` subnormal (`2^-9`) to zero and decoding a flat gray frame (`PSNR = 10.2 dB`). Additionally, DiffSynth 2.1.9's `VRAM_MANAGEMENT_MODULE_MAPS["diffsynth.models.qwen_image_21_controlnet.QwenImage21ControlNet"]` omits `QwenImage21ZeroCenterRMSNorm`. `_build_qwen21_vram_config` keeps the 7B `Qwen-Image-2.1` stack in `torch.bfloat16` (`cuda` when resident, `cpu` -> `cuda` when `--cpu-offload` is enabled) and registers `QwenImage21ZeroCenterRMSNorm` in `VRAM_MANAGEMENT_MODULE_MAPS`.

### 4. Licensing and integration decision

| Component | Repository | License | Commercial use | Role in `remove-ai-watermarks` |
| --- | --- | --- | --- | --- |
| Default Qwen global stage | `Qwen/Qwen-Image-2512` | **Apache-2.0** | Permitted | Default `qwen-zimage` and `auto` fallback |
| Shared face stage | `Tongyi-MAI/Z-Image-Turbo` | **Apache-2.0** | Permitted | Shared across all two-stage profiles |
| Qwen 2.1 ControlNet Union | `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` | **Apache-2.0** | Permitted | Used by opt-in `qwen21-zimage` |
| Qwen 2.1 Texture-Fix VAE | `madebyollin/texture-fix-vae-for-qwen-image-2.1` | **Apache-2.0** | Permitted | Used by opt-in `qwen21-zimage` |
| Qwen 2.1 base transformer | `Qwen/Qwen-Image-2.1` | **Qwen Research License** | **Non-commercial research & personal use only** | Opt-in `--pipeline qwen21-zimage` only (never selected by `auto`) |
| Qwen 2.1 Lightning LoRA | `lightx2v/Qwen-Image-2.1-Lightning` | **Qwen Research License** | **Non-commercial research & personal use only** | Not used (unfused 4-step Canny outperforms it) |

Because `Qwen/Qwen-Image-2.1` is governed by the non-commercial Qwen Research
License, `qwen-zimage` (`Qwen/Qwen-Image-2512`, Apache-2.0) remains the default
profile and `--pipeline auto` routes only across the Apache-2.0 / commercial-safe
profiles (`sdxl-zimage`, `chroma-zimage`, and `qwen-zimage`). `qwen21-zimage`
(`Qwen21ZImagePipeline` in `qwen_zimage_pipeline.py`) is shipped as an explicit
opt-in profile (`--pipeline qwen21-zimage`) with non-commercial / personal-use
notices in the CLI help, Python docstrings, documentation, and Agent Skill.
