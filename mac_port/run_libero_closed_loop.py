"""Closed-loop trials of one LIBERO task against a running GR00T server.
Each trial runs upstream's rollout client once with one seed."""
import ast
import json
import math
import os
import pathlib
import re
import socket
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
GR00T = HERE.parent
PY = GR00T / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
OUT = HERE / "runs" / "closed_loop_local"

TASK = "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 720
ACTION_STEPS = 8
HOST, PORT = "127.0.0.1", 5555
TIMEOUT_S = 600

PINS = {
    "gr00t_commit": "3b49067943eb1ab2eb2203b93d461c4390c78da3",
    "checkpoint": "nvidia/GR00T-N1.7-LIBERO/libero_10",
    "checkpoint_revision": "658540327860f93dd2cb9c1ed3e3b10fb792a0eb",
    "embodiment_tag": "LIBERO_PANDA",
    "device": "mps",
    "deviations_from_upstream": [
        "server: scipy upgraded past the pinned 1.15.3",
        "server: float32 instead of bfloat16 in gr00t/policy/gr00t_policy.py",
        "server: flash-attn absent, sdpa fallback",
        "sim: egl-probe skipped",
        "sim: mujoco plugin folder disabled",
        "sim: MUJOCO_GL=cgl, PYOPENGL_PLATFORM=darwin",
    ],
}


def trial(seed):
    out = OUT / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), "gr00t/eval/rollout_policy.py",
        "--n-episodes", "1", "--n-envs", "1",
        "--policy-client-host", HOST, "--policy-client-port", str(PORT),
        "--max-episode-steps", str(MAX_STEPS), "--n-action-steps", str(ACTION_STEPS),
        "--env-name", TASK, "--seed", str(seed), "--video-dir", str(out),
    ]
    result = {"task_id": TASK, "seed": seed, "success": False, "video_dir": str(out)}
    start = time.time()
    try:
        run = subprocess.run(
            cmd, cwd=GR00T, capture_output=True, text=True, timeout=TIMEOUT_S,
            env={**os.environ, "MUJOCO_GL": "cgl", "PYOPENGL_PLATFORM": "darwin"},
        )
    except subprocess.TimeoutExpired:
        return {**result, "termination": "timeout"}
    wall = time.time() - start
    text = run.stdout + run.stderr
    (out / "client.log").write_text(text)
    found = re.search(r"^results:\s*(.+)$", text, re.M)
    if run.returncode or not found:
        return {**result, "termination": "error"}
    _, (success,), info = ast.literal_eval(found.group(1))
    steps = info["episode_lengths"][0]
    return {
        **result,
        "success": bool(success),
        "termination": "success" if success else "max_steps" if steps >= MAX_STEPS else "env_terminated",
        "steps": steps,
        "wall_s": round(wall, 1),
        "latency_s_per_call_upper_bound": round(wall / math.ceil(steps / ACTION_STEPS), 2),
    }


if __name__ == "__main__":
    base = subprocess.run(["git", "-C", str(GR00T), "merge-base", "--is-ancestor", PINS["gr00t_commit"], "HEAD"])
    assert base.returncode == 0, f"checkout does not contain upstream {PINS['gr00t_commit']}"
    socket.create_connection((HOST, PORT), timeout=5).close()
    trials = []
    for seed in SEEDS:
        trials.append(trial(seed))
        print(seed, trials[-1]["termination"], flush=True)
    report = {"mode": "closed_loop", **PINS, "successes": sum(t["success"] for t in trials), "trials": trials}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{report['successes']}/{len(trials)} succeeded")
