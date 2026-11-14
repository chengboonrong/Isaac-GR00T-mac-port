# Fleet Wall Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a 3×2 "fleet wall" video of 6 LIBERO-10 pick/sort workcells × 3 jobs each, driven by GR00T N1.7 on MPS, with a jobs-completed counter.

**Architecture:** `mac_port/run_fleet_demo.py` runs the 18 jobs one after another through the existing `trial()` (refactored to take the task and output dir). It then composites the recorded clips frame by frame with numpy and Pillow and writes the result with imageio. Pure timeline and tiling helpers carry the logic and have one small test file.

**Tech Stack:** Python 3 in `gr00t/eval/sim/LIBERO/libero_uv/.venv` (imageio 2.38 + imageio-ffmpeg, Pillow 13, numpy 1.26), Homebrew ffmpeg for the GIF, and the font `/System/Library/Fonts/HelveticaNeue.ttc`.

**Spec:** `docs/superpowers/specs/2026-11-14-fleet-demo-design.md`

## Global Constraints

- Tasks are LIBERO-10 indices 0, 1, 4, 5, 6, 7, with env names `libero_sim/<task name>`. Seeds 0, 1, 2.
- Caption text, exactly: `6 simulated workcells · one GR00T N1.7 model · one MacBook (MPS)`.
- No picks-per-minute figure and no claim of real-time simultaneity anywhere: not in the video, the README or the report.
- The counter counts successes only and reads `Jobs completed: k / 18`.
- Don't change what `run_libero_closed_loop.py` does when it runs on its own.
- Code style: ruff format, line length 100.

## Review Focus

- A failed or timed-out job produces no mp4, or an mp4 with no `_s1` suffix. That cell should still play whatever clip exists, and if none exists, a black 512×284 slate of 20 frames. It must not crash. Test: `test_missing_clip_is_slate`.
- A failed job must not increment the counter, and the final counter value must equal `report["successes"]`. Test: `test_counter_skips_failures`.
- Cells have different total lengths. Shorter cells hold their last frame, and the video runs to the longest cell plus a 40-frame hold. Test: `test_shorter_cell_holds_last_frame`.
- The policy server isn't running. The script should fail fast with a clear message from the existing socket check before any job starts. Covered by reusing the check; no new test.
- If an output dir is left over from a previous run, stale mp4s from that run could get picked up. `trial()` should clear `out` before running. Covered in Task 1, Step 3.

---

### Task 1: Timeline and tiling helpers, plus `trial()` parameters

**Files:**
- Modify: `mac_port/run_libero_closed_loop.py` (`trial`)
- Create: `mac_port/fleet_compose.py`
- Test: `mac_port/test_fleet_compose.py`

**Interfaces:**
- Produces:
  - `trial(seed: int, task: str = TASK, out_root: pathlib.Path = OUT) -> dict`. Output goes in `out_root / f"seed_{seed}"`. The returned dict keeps the existing keys.
  - `completed_at(cells: list[list[tuple[int, bool]]], frame: int) -> int`. `cells[i]` is that cell's jobs as `(n_frames, success)`. It returns the number of successful jobs whose cumulative end frame is ≤ `frame`.
  - `cell_frame(clips: list[np.ndarray], frame: int) -> np.ndarray`. `clips` is the cell's job clips concatenated as `(T, H, W, 3)` uint8 arrays. Returns frame `frame`, clamped to the last frame.
  - `load_clip(video_dir: pathlib.Path) -> np.ndarray`. Reads the first `*.mp4` in the dir. If there isn't one, returns a black `(20, 284, 512, 3)` slate.
  - `tile(frames: list[np.ndarray], cols: int) -> np.ndarray`. Tiles them row-major into one image.

- [ ] **Step 1: Write the failing tests** in `mac_port/test_fleet_compose.py`. Use plain asserts, runnable as `python mac_port/test_fleet_compose.py`:

```python
def test_counter_skips_failures():
    cells = [[(10, True), (5, True)], [(8, False), (4, True)]]
    assert [completed_at(cells, t) for t in (9, 10, 12, 15, 99)] == [0, 1, 2, 3, 3]

def test_shorter_cell_holds_last_frame():
    clip = np.arange(3).reshape(3, 1, 1, 1).repeat(3, -1).astype(np.uint8)
    assert cell_frame([clip], 2)[0, 0, 0] == 2 and cell_frame([clip], 50)[0, 0, 0] == 2

def test_missing_clip_is_slate(tmp=pathlib.Path(tempfile.mkdtemp())):
    s = load_clip(tmp)
    assert s.shape == (20, 284, 512, 3) and s.max() == 0

def test_tile_shape():
    f = np.zeros((284, 512, 3), np.uint8)
    assert tile([f] * 6, cols=3).shape == (568, 1536, 3)
```

- [ ] **Step 2: Run** `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python mac_port/test_fleet_compose.py`. Expected: ImportError, because `fleet_compose` doesn't exist yet.
- [ ] **Step 3: Implement.**
  - Write the four helpers in `mac_port/fleet_compose.py`. `load_clip` uses `imageio.v3.imread(path)`.
  - In `trial()`, add the parameters `task` and `out_root`. Have it `shutil.rmtree(out, ignore_errors=True)` before `mkdir`.
- [ ] **Step 4: Run the tests again.** Expected: 4 passes. Also run `python3 -c "import ast;ast.parse(open('mac_port/run_libero_closed_loop.py').read())"`.
- [ ] **Step 5: Commit** with `git add mac_port/fleet_compose.py mac_port/test_fleet_compose.py mac_port/run_libero_closed_loop.py && git commit -m "Add fleet demo compositing helpers; parametrise trial()"`.

### Task 2: Fleet runner and render

**Files:**
- Create: `mac_port/run_fleet_demo.py`
- Modify: `mac_port/README.md` (add a "Fleet demo" section)
- Output (committed): `mac_port/results/fleet_demo.mp4`, `mac_port/results/fleet_demo.gif`, `mac_port/results/fleet_report.json`

**Interfaces:**
- Consumes: everything Task 1 produces, plus `PINS`, `HOST` and `PORT` from `run_libero_closed_loop.py`.
- Produces: `main()`. It runs the jobs, writes `mac_port/runs/fleet/report.json`, renders the outputs and copies them to `mac_port/results/`.

- [ ] **Step 1: Write `run_fleet_demo.py`.**
  - `TASKS` is a list of `(label, env_name)` for the 6 indices. The labels are short, e.g. `"Soup + sauce → basket"`.
  - Jobs run in task-major order through `trial(seed, task, OUT / slug)`.
  - Compose the frames:
    - The title bar is 1536×64: the caption at left and `Jobs completed: k / 18` at right.
    - Each cell gets its label drawn in its top-left corner.
    - The video is 20 fps and runs to the longest cell plus a 40-frame hold.
  - Write the mp4 with `imageio.v3.imwrite(..., fps=20, codec="libx264")`.
  - Make the GIF with `ffmpeg -i fleet_demo.mp4 -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" fleet_demo.gif`.
  - The report contains `**PINS`, `"successes"`, `"jobs": 18` and the trials.
  - Add a final check: `assert completed_at(cells, last_frame) == report["successes"]` and `assert len(TASKS) == 6`.
- [ ] **Step 2: Start the server** with the README's Terminal 1 command. Run in the background: `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python mac_port/run_fleet_demo.py`. Expected: 18 status lines, then `k/18 succeeded`, with `mac_port/results/fleet_demo.mp4` present.
- [ ] **Step 3: Verify the pixels.** Extract frames at 0 s, the midpoint and the end with `ffmpeg -ss <t> -frames:v 1`, and Read them. Check:
  - 6 distinct scenes are visible.
  - The labels are legible and the caption matches exactly.
  - The counter at the end equals `successes`.
- [ ] **Step 4: Write the README section.** Embed the GIF, the exact caption, a per-task success table taken from `fleet_report.json`, and the run command. Don't include any throughput claim.
- [ ] **Step 5: Commit** with `git add mac_port/run_fleet_demo.py mac_port/README.md mac_port/results/fleet_* docs/superpowers && git commit -m "Add fleet wall demo: 6 LIBERO-10 pick/sort workcells on MPS"`.
