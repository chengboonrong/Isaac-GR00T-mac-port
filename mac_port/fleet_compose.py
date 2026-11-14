"""Frame-level helpers for the fleet wall video: per-cell timelines, slates and tiling."""

import pathlib

import imageio.v3 as iio
import numpy as np


SLATE = (20, 284, 512, 3)


def completed_at(cells, frame):
    """Successful jobs whose cumulative end frame is <= frame. cells[i] = [(n_frames, success)]."""
    done = 0
    for jobs in cells:
        end = 0
        for n, success in jobs:
            end += n
            done += success and end <= frame
    return done


def cell_frame(clips, frame):
    """Frame `frame` of the clips played back to back, holding the last frame."""
    for clip in clips:
        if frame < len(clip):
            return clip[frame]
        frame -= len(clip)
    return clips[-1][-1]


def load_clip(video_dir):
    """First mp4 in video_dir, or a black slate when the job left no video."""
    videos = sorted(pathlib.Path(video_dir).glob("*.mp4"))
    return iio.imread(videos[0]) if videos else np.zeros(SLATE, np.uint8)


def tile(frames, cols):
    rows = [np.concatenate(frames[i : i + cols], axis=1) for i in range(0, len(frames), cols)]
    return np.concatenate(rows, axis=0)
