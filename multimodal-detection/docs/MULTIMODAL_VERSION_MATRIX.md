# Multimodal version matrix

| Version | Base route | Added mechanism | Known outcome / issue |
| --- | --- | --- | --- |
| V4.4 | Learned common/private fusion + memory | Independent auxiliary pretraining and joint fusion | Protected baseline; local best 0.456662. |
| V4.5 | Replaced deployment route | Spatial evidence router | Weakened embedding role; RGB-anchored weak residual did not improve. |
| V4.6 | V4.4 plus incremental router | More auxiliary injection | IR was not physically aligned; extra information did not improve quality. |
| V4.7 | Frozen V4.4 plus trusted-evidence router | Artificial IR affine supervision and multi-gated evidence | Best 0.456812 at ep2, final 0.4509; more injection correlated with lower AP. |
| V4.8 | Frozen V4.4 protected base | Real-pair rotation correction, embedding IR complement, depth geometry support, downstream low-LR adaptation | Current experiment. See `docs/V48_PLAN.md`. |

Every version has a distinct `fusion_strategy`, config file and run directory.
Never resume a checkpoint across strategies; use `--init-checkpoint` for explicit
migration, and `--resume` only inside the same run and exact recipe.

