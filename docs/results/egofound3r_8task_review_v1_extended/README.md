# Extended MANO eight-task visual review

This version keeps the original four-mode 2x2 visualization on the left, adds
two canonical MANO panels in the middle, and shows the original RGB video on
the right. The compact output layout is 2280x540 at 10 FPS:

```text
┌──────────────────┬──────────────┬──────────────────┐
│ Geometry │ Vis.  │ MANO Contact │                  │
│──────────┼───────│──────────────│   Original RGB   │
│ Contact  │ Dist. │ MANO Distance│                  │
└──────────────────┴──────────────┴──────────────────┘
```

Each canonical MANO panel contains the left hand on the left and right hand on
the right. The palm plane faces the viewer, the mean direction of the four
non-thumb fingertips points upward, and the thumbs point toward the layout
center. Contact and metric distance colors use the same legend as the 2x2
panels. The mesh is the interpolated 778-vertex MANO topology.

Outputs:

- [Guocai Ego](guocai_8tasks_extended.mp4)
- [Jianzhi Ego](jianzhi_8tasks_extended.mp4)
- [Stera-10M](stera_8tasks_extended.mp4)
- [All three datasets](all_three_24tasks_extended.mp4)
