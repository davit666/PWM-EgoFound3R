# EgoFound3R eight-task visual review

Each dataset contributes eight distinct tasks. Every selected task is clipped at
its temporal center to 8 seconds, normalized to 10 FPS, and concatenated in task
order. The published videos contain the four paper-style prediction modes using
the model's native 195-marker output.

| Dataset | Candidate tasks | Selected tasks | Published video | Selection details |
|---|---:|---:|---|---|
| Guocai Ego | 294 | 8 | [guocai_8tasks.mp4](guocai_8tasks.mp4) | [manifest](guocai_manifest.json) |
| Jianzhi Ego | 100 | 8 | [jianzhi_8tasks.mp4](jianzhi_8tasks.mp4) | [manifest](jianzhi_manifest.json) |
| Stera-10M | 575 | 8 | [stera_8tasks.mp4](stera_8tasks.mp4) | [manifest](stera_manifest.json) |

The three published videos are each 64 seconds, 640 frames, 1280x720 H.264.
The combined [all_three_24tasks.mp4](all_three_24tasks.mp4) is 192 seconds and
contains all 24 tasks.

Jianzhi stores six synchronized H.264 camera streams in every MCAP. The current
public inference entrypoint accepts one RGB video and exposes no physical-view
axis or per-view calibration interface, so this run uses the forward-facing
`/robot0/sensor/camera2/compressed` stream. The
[six-camera contact sheet](jianzhi_camera_preview/camera_contact_sheet.jpg) is
included for later native multi-view development; the camera streams were not
tiled into a pseudo multi-view input.
