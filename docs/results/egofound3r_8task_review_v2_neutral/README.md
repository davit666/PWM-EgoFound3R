# Fixed-neutral MANO eight-task visual review

This corrected version keeps the original four-mode 2x2 visualization on the
left, adds two fixed-neutral MANO panels in the middle, and shows the original
RGB video on the right. The compact output layout is 2280x540 at 10 FPS:

```text
┌──────────────────┬──────────────┬──────────────────┐
│ Geometry │ Vis.  │ MANO Contact │                  │
│──────────┼───────│──────────────│   Original RGB   │
│ Contact  │ Dist. │ MANO Distance│                  │
└──────────────────┴──────────────┴──────────────────┘
```

The middle panels do **not** render the predicted hand pose. Their geometry is
the fixed left/right neutral MANO template from the PaXini tactile
representation calibration assets. Only the per-frame 778-vertex contact and
metric-distance values change. Both palms face the viewer, the four non-thumb
fingertips point upward, the left hand stays on the left, and the right hand
stays on the right.

Outputs are published by `scripts/pwm/run_extended_mano_review.sh` after full
generation and decode validation.
