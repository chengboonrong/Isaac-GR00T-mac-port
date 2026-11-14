# Fleet wall demo: design

## Intent
A simple, cheap demo for a client presentation: one GR00T N1.7 model running a "fleet" of robot arm workcells that pick and sort objects, all on one Mac. Delivered as a recorded video, so nothing can fail during the meeting.

## Goals
- One grid video of 6 workcells, each doing a different LIBERO-10 pick/sort task.
- Each workcell runs 3 jobs back to back, so the video lasts about 15–20 s and the job count keeps going up.
- An overlay with the title, each cell's task, and a "jobs completed: k / 18" counter that goes up as jobs finish.
- A small `report.json` with the real outcome of each job.

## Non-goals
- Live running in the meeting, mobile robots or a warehouse floor, new tasks, fine-tuning, a web dashboard.

## Anti-goals
- Overclaiming. The jobs run one after another and are tiled afterwards, so the caption says "6 simulated workcells · one GR00T model · one MacBook". It does not say they ran in real time at once, and it does not give a picks-per-minute figure.
- Hiding failures. If a job fails, the counter counts only successes and the report says so. Whether to rerun with another seed is the user's call.

## Approaches
- **A. Recorded fleet wall video (picked):** cheapest, and nothing can go wrong in the room.
- B. The same, run live: impressive, but one shared MPS server at about 0.85 s per call stutters, and a hang hurts.
- C. A warehouse with mobile robots: GR00T can't drive them, so they would be scripted. Much more work, and mostly not GR00T.

## Design
- **Tasks** (LIBERO-10 indices 0, 1, 4, 5, 6, 7, all pick/place or sort):
  soup+sauce → basket · cream cheese+butter → basket · mugs sorted onto left/right plates · book → caddy · mug → plate + pudding to the right · soup+cream cheese → basket.
- **Runner** `mac_port/run_fleet_demo.py`: reuses `trial()` from `run_libero_closed_loop.py` (made to take the task and output dir as arguments). It runs 6 tasks × seeds 0–2 one after another against the running MPS server and writes `mac_port/runs/fleet/<task>/seed_<n>/` plus `report.json`.
- **Compositor**, in the same script, using imageio, numpy and Pillow from the LIBERO venv (Homebrew's ffmpeg has no `drawtext`): each cell plays its 3 job clips (512×284 @ 20 fps) back to back and holds its last frame. The cells are tiled 3×2 under a title bar, Pillow draws the labels, and the counter steps up on the frame where each successful job ends. Output: `mac_port/results/fleet_demo.mp4`, plus a GIF made with ffmpeg `palettegen` for the README.
- **Check**: the script asserts that the grid has 6 cells and that the counter's final value equals the number of successes in the report.

## Left to the builder
Font size, label colors and wording, how long the clip ends hold on the last frame.
