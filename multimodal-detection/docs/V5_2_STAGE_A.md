# V5.2 Stage A

V5.2 starts exclusively from the V4.4 \`best.pt\` checkpoint. V4.5 through V5.1
remain experimental history and their fusion policies are not inherited.

## Scope

- Train RGB, IR, and depth as independent detection branches during Stage A.
- Keep the RGB detector frozen as the protected V4.4 reference.
- Read IR as the median of its three channels.
- Apply the accepted A0 coarse rotation only while sampling IR features; never
  overwrite the raw IR image.
- Let the learned IR residual module absorb remaining local translation and
  scale error.
- Exclude confirmed cross-modal mismatches through a versioned stem list.
- Do not launch Stage B automatically.

## Geometry contract

The A0 cache contains an image-derived coarse rotation candidate and quality
metadata. It is an input option, not a ground-truth affine label. Cached
rotation is transported through crop, flip, and letterbox transforms before
feature resampling. Mosaic and sensor-only affine perturbations are therefore
disabled for this recipe.

The IR independent branch does not read RGB semantic features. RGB may define
the annotation coordinate system and offline geometric calibration, but it
cannot supply appearance evidence to the IR detector.

## Stage B gate

Stage B remains manual. It may be considered only after independent validation
shows that IR reaches the agreed threshold, RGB remains protected, cached
geometry is valid, and confidence correlates with actual alignment improvement.

## Reproducibility tools

- \`tools/build_v52_rotation_cache.py\`: creates the versioned rotation-input
  cache without depending on historical probe scripts.
- \`tools/check_v52_entrypoint.py\`: verifies that direct entry points import the
  repository-local dataset and model modules.
- \`tools/preflight_v52.py\`: performs checkpoint, geometry, modality-isolation,
  optimizer-step, and checkpoint-roundtrip checks on the server.
- \`tools/launch_v52_stage_a.sh\`: records the full Stage A launch command.
