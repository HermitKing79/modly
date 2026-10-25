# Zombie input-quality preparation: NO-GO for a tonal generation A/B

Base `de1cd7848976a34a918c9edc28473a0b34ea7c1c`. One bounded CPU image experiment; the original image, geometry and neural evidence remain unchanged. No neural imports, model loading, WSL, inference, attempts or generation-setting changes.

**Decision:** keep the original reference. The one shadow-lift candidate is faithful but adds no useful anatomical separation or cavity information. Do not spend a neural attempt on it. Obtain a better source reference before proposing an input-only generation A/B. This result concerns this image and candidate, not every possible reference or preprocessing method.

## Authentication and actual input

The original Downloads PNG, preserved snapshot and original DINO scratch input match SHA-256 `db1fbb6bd27717f4b6a0298d64ff1ddff17e80027d782bf5dbfbc959d70fa06e` (1216×1294 RGBA). The saved conditioning file still hashes to `e5659b1842509e73ec496ef90bf95fae86c93ddf5ee73fba8015822982a3cb60`; its header names that same image.

Extracting only the pinned CPU `preprocess_image` method, with RMBG absent, reproduces the archived 980×980 RGB crop exactly: pixel SHA-256 `655ef5f8144975213ba5ce41e684473b9c0690f4b661561b74282c7ba949f92e`. The extractor's documented-in-source Lanczos resize to 512×512 gives pixel SHA-256 `f5607215b0c21f8fc487bb321be5732b7c47e357a4b66ee48843948e8250b29c`. This is a deterministic reconstruction of the pre-normalization raster, not a newly executed DINO pass or a historically saved 512-pixel receipt. No model modules were imported.

## What the model can see

| Feature | Observation at effective 512 resolution |
|---|---|
| Hands | Edge-on fingers overlap into one broad projection plus a thumb. Individual hidden fingers are absent from the source view, not recoverable by sharpening. Inspection crops are 61×31 and 62×31 pixels including context. |
| Gaps | At image x=35 and x=477, thumb-gap samples are 2–4 pixels wide across alpha thresholds 32/128/204. These are local raster measurements, not whole-gap minima or anatomical dimensions. |
| Toes | Toe forms and dark grooves are visible. At sampled rows 500/503/506, each forefoot remains a single alpha foreground interval. A one-pixel interruption at left row 509 appears at alpha >204 but disappears at >128: little robust negative space. |
| Face | Eyes, nose, mouth and teeth are readable in a 62×64-pixel context crop. Fine teeth/eye boundaries have few pixels; the source's strong faceted shading remains. |
| Wounds | Ragged rims and reddish interiors remain visible in chest 38×42 and abdomen 31×53 context crops. The source does not clearly disambiguate cavity walls, depth or backside geometry. Dark color alone cannot supply those constraints. |

The mask retains the wounds as foreground rather than cutting them out. Foreground source alpha percentiles 10/50/90 are 252/253/253: near-opaque, not large missing regions. At the pipeline threshold under four-neighbor connectivity, the 512 mask contains one main 47,893-pixel component, seven single-pixel speckles and one enclosed three-pixel background region near the image-left face edge. These raster counts do not describe 3D topology or establish mask error; none was automatically repaired.

The thresholded 512 foreground bounding box is `[11,1,501,512)`; 21 foreground pixels touch the bottom row. Framing is already tight. Extra transparent padding would be removed by the same automatic crop; enlarging only hands/face would change proportions or discard the rest of the character.

## One candidate, measured and rejected

Apply a monotonic RGB lookup `round(255*(v/255)^0.85)` to the original source, leaving alpha bit-exact. No painting, spatial filtering, sharpening, resizing of the source or invented anatomy. Reprocess with the original method/settings.

- Alpha changed pixels: **0**. Conditioning silhouette XOR: **0**; IoU **1.0**. Gap widths unchanged.
- Face dark-decile sRGB-code luma rises **35.6 → 47.1**, while the 90th-minus-10th percentile range falls **154.3 → 150.6**. Chest context: **81.5 → 94.8**, range **119.3 → 113.0**. These describe tonal redistribution, not proven perceptual or generation improvement.
- Matched full-size and enlarged views show brighter shadows but no new finger separations, toe gaps or cavity-wall evidence. Faceted shading remains. The candidate does not meet the practical improvement criterion.
- The candidate/comparison stage took **0.77 seconds**, peaked at **97,726,464 bytes working set**, and ended at **573,112,320 bytes private commit**. Sampled available RAM stayed above the existing 12 GiB CPU live floor; the existing 6 GiB commit, 2 GiB disk and 180-second stage bounds passed. This was not neural admission testing.

Private evidence leaf `amd-zombie-input-quality-20261025-v1`: `comparison-512.png` shows A/B at actual 512 pixels; `comparison-details-v2.png` shows matched 4× nearest-pixel crops; `alpha-and-feature-locations.png` shows the mask and context boxes. The first `comparison-details.png` had layout overlap and is retained as superseded; use v2. Images, candidate reference, pixel measurements and reproduction scripts stay local.

**Next action:** request a faithful same-character source showing distinct fingers/toes and clearly lit wound rims and sidewalls at the eventual 512 framing. This requires additional visible source information, not fabricated anatomy. Do not substitute a collage or assume unqualified multi-view inference. Once a source demonstrably improves these cues, separately propose one input-only generation A/B against the existing zombie; retain original sampler/checkpoints/decoder and every safety gate. No neural A/B is proposed for the rejected tonal candidate, and none is authorized or started.
