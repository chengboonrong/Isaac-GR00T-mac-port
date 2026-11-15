"""Same libero_10 checkpoint on LIBERO tasks it was NOT fine-tuned on (goal/spatial/object suites).
Same robot, simulator, seeds and 720-step limit as run_fleet_demo.py, so the two reports compare
trained vs untrained tasks directly. Run with the LIBERO venv python while the MPS server is up."""

import json
import shutil
import socket
import sys

from run_fleet_demo import SEEDS, render, run_jobs
from run_libero_closed_loop import HERE, HOST, PINS, PORT


# Simple single-object tasks, several with objects the trained tasks also use (soup, sauce, cheese).
TASKS = [
    ("Soup → basket", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    ("Tomato sauce → basket", "pick_up_the_tomato_sauce_and_place_it_in_the_basket"),
    ("Bowl (centre) → plate", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"),
    ("Bowl (by ramekin) → plate", "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"),
    ("Bowl → plate", "put_the_bowl_on_the_plate"),
    ("Cream cheese → bowl", "put_the_cream_cheese_in_the_bowl"),
]  # fmt: skip
SUITES = ["libero_object"] * 2 + ["libero_spatial"] * 2 + ["libero_goal"] * 2
CAPTION = "Same GR00T model · tasks it was NOT trained on"
NOTE = "LIBERO object / spatial / goal tasks · each job recorded separately · not real time"
OUT = HERE / "runs" / "untrained"
RESULTS = HERE / "results"


def main(render_only=False):
    """--render-only rebuilds the video from runs/untrained/report.json without rerunning jobs."""
    if render_only:
        trials = json.loads((OUT / "report.json").read_text())["trials"]
    else:
        socket.create_connection((HOST, PORT), timeout=5).close()
        trials = run_jobs(TASKS, OUT)
        for t in trials:
            t["suite"] = SUITES[[label for label, _ in TASKS].index(t["label"])]
    assert len(trials) == len(TASKS) * len(SEEDS)
    successes = sum(t["success"] for t in trials)
    report = {
        "mode": "untrained_check",
        **PINS,
        "successes": successes,
        "jobs": len(trials),
        "trials": trials,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    RESULTS.mkdir(exist_ok=True)
    final = render(trials, RESULTS / "untrained_check.mp4", TASKS, CAPTION, NOTE)
    assert final == successes, "final counter disagrees with report"
    shutil.copy(OUT / "report.json", RESULTS / "untrained_report.json")
    print(f"{successes}/{len(trials)} succeeded")


if __name__ == "__main__":
    main(render_only="--render-only" in sys.argv)
