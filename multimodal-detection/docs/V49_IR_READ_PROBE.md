# V4.9 IR channel-reduction probe

## Question

Does replacing OpenCV channel 0 with the per-pixel median of the stored IR
channels remove weak chroma residue without damaging the thermal signal, and
does the network benefit after a short adaptation?

This is **not** a geometric-alignment experiment. It neither rotates nor warps
the original IR image.

## Validation ladder

1. Local unit tests verify exact legacy compatibility, median reduction,
   grayscale identity, uint16 scaling, invalid-mode rejection, and checkpoint
   round-tripping.
2. A local contact sheet checks that thermal structure, black borders, and
   apparent tilt remain unchanged.
3. Server zero-shot A/B evaluates the same V4.4 `best.pt` on the fixed 400-image
   split with only `ir_read_mode` changed.
4. A paired six-epoch server probe starts both arms from the same V4.4
   `best.pt`, using the same seed, batches, augmentation, optimizer recipe, and
   validation split. The only intended difference is `ir_read_mode`.

## Decision rule

Adopt `median_channel` for the next fusion model only if its paired short-run
best mAP50-95 exceeds the legacy arm by at least 0.002, or if the global metric
is tied while IR-sensitive classes and the `no_ir`/`no_rgb` profile improve
without a material class regression. A larger IR route or auxiliary signal is
not sufficient when detection metrics decline.

Keep all V4.4/V4.8 weights and logs. Do not run a full refit from this probe.
