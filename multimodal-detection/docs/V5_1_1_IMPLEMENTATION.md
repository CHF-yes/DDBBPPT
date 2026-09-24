# V5.1.1 implementation contract

V5.1.1 uses the repository tag `v4.4` and its `best.pt` as the only detector
baseline.  V4.5--V5 are diagnostic history, not a weight or routing baseline.

The complete design authority is `_temp_v5_1_1方案.md` in the project
workspace.  This file records the executable contract so that training cannot
silently drift from that design.

## A0 cache

`tools/build_ir_a0_cache.py` is an offline, two-pass analysis job:

1. read the median IR intensity without modifying the source image;
2. detect border-connected invalid/dark regions and produce separate visible
   and conservative geometry masks;
3. estimate RGB-projection leakage and thermal-structure alignment separately;
4. robustly aggregate a sequence/source affine prior;
5. estimate a bounded per-image residual around that prior;
6. emit confidence, supervision validity and low-resolution quality maps;
7. emit visual audit sheets and dataset-level statistics.

The stored affine is a **source-sampling transform**: for an RGB/reference
output coordinate it returns the IR input coordinate that should be sampled.
The raw IR file is never overwritten.  Weak or conflicting evidence produces
`affine_supervised=0`; it never produces a fabricated precise transform.
Only a named source with at least two supporting images may share a sequence
prior.  `PLAIN` and singleton images retain their own coarse-to-fine estimate;
they never inherit a dataset-wide transform.  An unsupervised cached affine is
also excluded from known synthetic-affine target composition.
The cache writer additionally conjugates every candidate through the declared
training canvas and marks transforms outside the aligner's angle/shift/scale
range as `affine_supervised=0`.  Their correlation confidence remains available
to the quality path, but they cannot contaminate geometric or synthetic labels.
The production recipe uses a bounded four-degree angle range.  This preserves a
small margin above the observed high-confidence three-degree search boundary;
the translation and scale limits remain conservative and are checked
independently.

## Stage A

Stage A trains RGB and IR standalone detectors and the IR alignment/quality
modules.  RGB is a geometry teacher only; the IR detector never reads RGB
features.  Cached pseudo labels and known synthetic transforms supervise the
affine head.  Stage B is blocked unless IR AP, synthetic recovery, affine
dispersion, cache coverage and mask audit gates pass.

## Stage B

The complete learned V4.4 detector route is the protected anchor.  V5.1.1 adds
zero-initialized spatial/channel residuals.  Shared IR evidence, thermal-private
evidence and processing-chain quality remain distinct.  RGB leakage is a
redundancy/quality signal, never independent thermal evidence.  Invalid border
pixels are hard-masked.  Low-confidence IR may help objectness/classification
but cannot perturb the P2/P3 localization path.

## Safety and compatibility

- checkpoints remain self-describing;
- missing A0 cache fails closed when the recipe requires it;
- inference may run without A0 only through the identity/raw-IR fallback;
- V4.4 checkpoints retain their original three-channel quality contract;
- original images, labels and historical weights are never modified.
