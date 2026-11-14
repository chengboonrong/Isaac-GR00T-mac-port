"""Fleet wall demo: 6 LIBERO-10 pick/sort workcells x 3 jobs, one GR00T server, one grid video.
Jobs run one after another; the grid is assembled afterwards from the recorded clips.
Run with the LIBERO venv python while the MPS policy server is up (see README)."""

import json
import pathlib
import shutil
import socket
import subprocess
import sys

from fleet_compose import cell_frame, completed_at, load_clip, tile
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run_libero_closed_loop import HERE, HOST, PINS, PORT, trial


TASKS = [
    ("Soup + sauce → basket", "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"),
    ("Cream cheese + butter → basket", "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"),
    ("Sort mugs → left / right plate", "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"),
    ("Book → caddy", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"),
    ("Mug → plate, pudding → right", "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"),
    ("Soup + cream cheese → basket", "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"),
]  # fmt: skip
SEEDS = [0, 1, 2]
COLS, FPS, HOLD, BAR = 3, 20, 40, 80  # 512 + 80 = 592, a multiple of 16 for x264
CAPTION = "6 simulated workcells · one GR00T N1.7 model · one MacBook (MPS)"
NOTE = "Each job recorded separately and tiled afterwards · not real time"
FONT = "/System/Library/Fonts/SFNS.ttf"
OUT = HERE / "runs" / "fleet"
RESULTS = HERE / "results"


def run_jobs():
    trials = []
    for label, name in TASKS:
        for seed in SEEDS:
            t = trial(seed, f"libero_sim/{name}", OUT / name)
            rel = str(
                pathlib.Path(t["video_dir"]).relative_to(HERE)
            )  # no local paths in the report
            trials.append({**t, "video_dir": rel, "label": label})
            print(label, seed, t["termination"], flush=True)
    return trials


def render(trials, path):
    cells, clips = [], []
    for i in range(len(TASKS)):
        jobs = trials[i * len(SEEDS) : (i + 1) * len(SEEDS)]
        cell_clips = [load_clip(HERE / j["video_dir"]) for j in jobs]
        clips.append(cell_clips)
        cells.append([(len(c), j["success"]) for c, j in zip(cell_clips, jobs)])
    n_frames = max(sum(n for n, _ in jobs) for jobs in cells) + HOLD
    total = len(trials)
    big, small, note = (ImageFont.truetype(FONT, size) for size in (28, 20, 17))
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", pixelformat="yuv420p")
    for f in range(n_frames):
        grid = Image.fromarray(tile([cell_frame(c, f) for c in clips], COLS))
        draw = ImageDraw.Draw(grid)
        cw, ch = grid.width // COLS, grid.height // (len(TASKS) // COLS)
        for i, (label, _) in enumerate(TASKS):
            x, y = (i % COLS) * cw + 10, (i // COLS) * ch + 8
            draw.text((x, y), label, font=small, fill="white", stroke_width=2, stroke_fill="black")
        canvas = Image.new("RGB", (grid.width, grid.height + BAR), (18, 18, 24))
        canvas.paste(grid, (0, BAR))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 28), CAPTION, font=big, fill="white", anchor="lm")
        draw.text((16, 62), NOTE, font=note, fill=(170, 170, 180), anchor="lm")
        counter = f"Jobs completed: {completed_at(cells, f)} / {total}"
        draw.text((canvas.width - 16, 28), counter, font=big, fill=(118, 214, 140), anchor="rm")
        writer.append_data(np.asarray(canvas))
    writer.close()
    return completed_at(cells, n_frames - 1)


def main(render_only=False):
    """--render-only rebuilds the video from runs/fleet/report.json without rerunning jobs."""
    assert len(TASKS) == 6
    if render_only:
        trials = json.loads((OUT / "report.json").read_text())["trials"]
    else:
        socket.create_connection((HOST, PORT), timeout=5).close()
        trials = run_jobs()
    successes = sum(t["success"] for t in trials)
    report = {
        "mode": "fleet_demo",
        **PINS,
        "successes": successes,
        "jobs": len(trials),
        "trials": trials,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    RESULTS.mkdir(exist_ok=True)
    mp4 = RESULTS / "fleet_demo.mp4"
    assert render(trials, mp4) == successes, "final counter disagrees with report"
    shutil.copy(OUT / "report.json", RESULTS / "fleet_report.json")
    gif_filter = (  # ~6 MB README preview; the mp4 is the presentation asset
        "fps=6,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:diff_mode=rectangle"
    )
    gif = RESULTS / "fleet_demo.gif"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vf", gif_filter, gif], check=True)
    print(f"{successes}/{len(trials)} succeeded")


if __name__ == "__main__":
    main(render_only="--render-only" in sys.argv)
