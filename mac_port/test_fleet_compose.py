"""Checks for the fleet demo compositing helpers. Run: python mac_port/test_fleet_compose.py"""

import pathlib
import tempfile

from fleet_compose import cell_frame, completed_at, load_clip, tile
import imageio.v3 as iio
import numpy as np


def test_counter_skips_failures():
    cells = [[(10, True), (5, True)], [(8, False), (4, True)]]
    assert [completed_at(cells, t) for t in (9, 10, 12, 15, 99)] == [0, 1, 2, 3, 3]


def test_shorter_cell_holds_last_frame():
    clip = np.arange(3).reshape(3, 1, 1, 1).repeat(3, -1).astype(np.uint8)
    assert cell_frame([clip], 2)[0, 0, 0] == 2
    assert cell_frame([clip], 50)[0, 0, 0] == 2


def test_cell_frame_crosses_into_next_clip():
    a, b = np.zeros((2, 1, 1, 3), np.uint8), np.full((2, 1, 1, 3), 7, np.uint8)
    assert cell_frame([a, b], 1)[0, 0, 0] == 0
    assert cell_frame([a, b], 2)[0, 0, 0] == 7


def test_missing_clip_is_slate():
    s = load_clip(pathlib.Path(tempfile.mkdtemp()))
    assert s.shape == (20, 256, 512, 3) and s.max() == 0


def test_load_clip_crops_overlay_strip():
    # Upstream burns "<task> (success flag)" into rows 256+; the flag often reads (0) on success.
    d = pathlib.Path(tempfile.mkdtemp())
    clip = np.zeros((4, 284, 512, 3), np.uint8)
    clip[:, 256:] = 255
    iio.imwrite(d / "x_s1.mp4", clip, fps=20, codec="libx264", macro_block_size=1)
    out = load_clip(d)
    assert out.shape == (4, 256, 512, 3) and out.max() < 40


def test_tile_shape():
    f = np.zeros((284, 512, 3), np.uint8)
    assert tile([f] * 6, cols=3).shape == (568, 1536, 3)


def test_tile_pads_mismatched_heights():
    # LIBERO scenes 5/6 record 512x280 clips, the rest 512x284.
    a, b = np.ones((284, 512, 3), np.uint8), np.ones((280, 512, 3), np.uint8)
    out = tile([a, b, a, a, b, a], cols=3)
    assert out.shape == (568, 1536, 3)
    assert out[280:284, 512:1024].max() == 0  # padding below the shorter cell is black


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
