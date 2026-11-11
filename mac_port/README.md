# GR00T N1.7 on Apple Silicon

This branch runs NVIDIA's `GR00T-N1.7-LIBERO` policy server and the LIBERO simulator on a Mac (PyTorch MPS), with no CUDA.

![Closed-loop rollout, seed 0](runs/closed_loop/seed_0.gif)

*`STUDY_SCENE1`: "pick up the book and place it in the back compartment of the caddy". Closed loop, MPS, seed 0.*

## Result

5 out of 5 closed-loop trials succeeded on one LIBERO-10 task.

| Seed | Result | Steps | Wall time | Video |
|---|---|---|---|---|
| 0 | success | 163 | 19.1 s | [seed_0.mp4](runs/closed_loop/seed_0.mp4) |
| 1 | success | 172 | 19.8 s | [seed_1.mp4](runs/closed_loop/seed_1.mp4) |
| 2 | success | 151 | 17.9 s | [seed_2.mp4](runs/closed_loop/seed_2.mp4) |
| 3 | success | 165 | 19.2 s | [seed_3.mp4](runs/closed_loop/seed_3.mp4) |
| 4 | success | 269 | 27.0 s | [seed_4.mp4](runs/closed_loop/seed_4.mp4) |

Each policy call takes about 0.8–0.9 s (upper bound: wall time ÷ action chunks, 8 actions per chunk). Full record: [`runs/closed_loop/report.json`](runs/closed_loop/report.json).

Pinned to upstream commit `3b49067` and checkpoint `nvidia/GR00T-N1.7-LIBERO/libero_10` @ `6585403`.

## What differs from upstream

| Where | Change |
|---|---|
| `gr00t/policy/gr00t_policy.py` | Model weights and inputs in float32 instead of bfloat16 (committed on this branch) |
| server env | `scipy` upgraded past the pinned 1.15.3 |
| server env | `flash-attn` not installed, attention falls back to SDPA |
| sim env | `egl-probe` skipped (no EGL on macOS) |
| sim env | MuJoCo plugin folder disabled |
| sim env | `MUJOCO_GL=cgl`, `PYOPENGL_PLATFORM=darwin` |

Only the float32 change is committed. The environment changes were made by hand in the local venvs.

## Reproduce

Set up the sim and download the checkpoint as in [`examples/LIBERO/README.md`](../examples/LIBERO/README.md), applying the environment changes above. Then:

```bash
# Terminal 1: policy server on MPS
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --embodiment-tag LIBERO_PANDA --device mps --use-sim-policy-wrapper

# Terminal 2: 5 seeded closed-loop trials, writes mac_port/runs/closed_loop_local/
python3 mac_port/run_libero_closed_loop.py
```
