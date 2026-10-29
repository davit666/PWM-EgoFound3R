"""Timing summaries use measured optimizer steps, never inference FPS."""
import math
import statistics


def summarize_step_times(records, *, discard_first=5, projected_steps=3200):
    seconds=[float(row['step_seconds']) for row in records]
    if not seconds or any(not math.isfinite(x) or x<=0 for x in seconds):
        raise ValueError('positive finite step timings required')
    kept=seconds[discard_first:] if len(seconds)>discard_first else seconds
    ordered=sorted(kept)
    mean=statistics.mean(kept)
    return {'measured_steps':len(seconds),'training_step_seconds_total':sum(seconds),
        'steady_steps':len(kept),'discarded_startup_steps':len(seconds)-len(kept),
        'steady_mean_step_seconds':mean,'steady_median_step_seconds':statistics.median(kept),
        'steady_p95_step_seconds':ordered[max(0,math.ceil(.95*len(ordered))-1)],
        'projected_steps':projected_steps,'projected_training_hours':mean*projected_steps/3600,
        'scope':'step time includes data loading and GPU work; excludes initialization, checkpoint saving and evaluation'}
