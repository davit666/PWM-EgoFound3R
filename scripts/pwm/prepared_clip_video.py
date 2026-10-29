#!/usr/bin/env python3
"""Encode the exact prepared RGB grid for an inference export smoke test."""
import sys
import subprocess
import numpy as np
with np.load(sys.argv[1],allow_pickle=False) as clip:
    rgb,time=clip['rgb'],clip['timestamp']
    fps=1/float(np.median(np.diff(time)))
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{rgb.shape[2]}x{rgb.shape[1]}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-crf','0','-pix_fmt','yuv444p',sys.argv[2]],input=rgb.tobytes(),check=True)
