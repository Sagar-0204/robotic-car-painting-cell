"""Step 1: plan the full painting program, verify it, and save results.

    python plan_program.py

Outputs (folder ./output):
    program.npz         time-stamped joint trajectory + paint deposition (used by viewer/video)
    results.json        all metrics (coverage, cycle time, clearance, joint speeds ...)
    plots/*.png         figures for the report
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paintcell import analysis, paths
from paintcell.cell import Cell, ARM_JOINTS
from paintcell.collision import SAFETY_MARGIN, CONTACT_MARGIN
from paintcell.planner import Planner, PAINT_SPEED

OUT = Path(__file__).resolve().parent / "output"
PLOTS = OUT / "plots"


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cell = Cell()

    print("== Planning painting program ==")
    planner = Planner(cell)
    prog = planner.build()
    print(f"   planned {len(prog['t'])} samples, cycle time {prog['t'][-1]:.1f} s "
          f"({time.time() - t0:.0f} s compute)")

    print("== Collision sweep over every sample ==")
    dmin, ncon, pairs = analysis.collision_sweep(cell, prog)
    i_min = int(np.argmin(dmin))
    print(f"   min clearance {dmin[i_min]*1000:.0f} mm ({pairs[i_min][0]} <-> {pairs[i_min][1]}), "
          f"contacts: {int((ncon > 0).sum())} samples")

    print("== Paint deposition + coverage ==")
    dots = analysis.deposit(cell, prog)
    film = analysis.film_thickness(cell, dots)
    cov = {k: v["coverage"] for k, v in film.items()}
    print("   surface      coverage  mean film  CV    within +-25%")
    for k, v in film.items():
        print(f"   {k:11s} {v['coverage']*100:6.1f} %   {v['mean']:5.2f}   {v['cv']:5.2f}   "
              f"{v['within25']*100:5.1f} %")

    speeds = analysis.joint_speed_ratio(prog)
    t = prog["t"]
    dt = np.diff(t, append=t[-1])
    kind = prog["kind"]
    paint_time = float(dt[kind == 3].sum())
    results = dict(
        cycle_time_s=round(float(t[-1]), 1),
        painting_time_s=round(paint_time, 1),
        conveyor_time_s=round(float(dt[kind == 0].sum()), 1),
        transfer_time_s=round(float(dt[(kind == 1) | (kind == 2)].sum()), 1),
        paint_speed_m_s=PAINT_SPEED,
        standoff_m=paths.STANDOFF, fan_width_m=paths.FAN_WIDTH, row_pitch_m=paths.ROW_PITCH,
        surfaces=planner.stats,
        coverage_pct={k: round(v * 100, 1) for k, v in cov.items()},
        total_coverage_pct=round(float(np.average(
            [v["coverage"] for v in film.values()],
            weights=[len(v["points"]) for v in film.values()])) * 100, 1),
        film_mean_rel={k: round(v["mean"], 3) for k, v in film.items()},
        film_cv={k: round(v["cv"], 3) for k, v in film.items()},
        film_within_25pct={k: round(v["within25"] * 100, 1) for k, v in film.items()},
        min_clearance_mm=round(float(dmin[i_min]) * 1000, 1),
        min_clearance_pair=list(pairs[i_min]),
        min_clearance_time_s=round(float(t[i_min]), 2),
        samples_below_50mm=int((dmin < SAFETY_MARGIN).sum()),
        samples_in_contact=int((ncon > 0).sum()),
        peak_joint_speed_ratio=speeds,
        rejected_motions=planner.events,
        mean_standoff_mm=round(float(dots[dots[:, 8] > 0, 7].mean()) * 1000, 1),
        standoff_std_mm=round(float(dots[dots[:, 8] > 0, 7].std()) * 1000, 2),
        overspray_m=paths.OVERSPRAY,
    )
    np.savez_compressed(OUT / "program.npz", **prog, clearance=dmin, dots=dots)
    # rejected (colliding) motions are saved too, for the collision demo in the video
    rej = planner.rejected
    if rej:
        np.savez(OUT / "rejected.npz", q0=np.array([r["q0"] for r in rej]),
                 q1=np.array([r["q1"] for r in rej]),
                 labels=np.array([r["label"] for r in rej]))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    make_plots(prog, dmin, dots, cov, film)
    print(f"== Done in {time.time() - t0:.0f} s -> {OUT}")
    print(json.dumps({k: results[k] for k in
                      ["cycle_time_s", "painting_time_s", "total_coverage_pct",
                       "min_clearance_mm", "samples_in_contact"]}, indent=2))


def make_plots(prog, dmin, dots, cov, film):
    t, q = prog["t"], prog["q"]
    spray = prog["spray"]

    # 1. joint trajectories
    fig, axes = plt.subplots(4, 2, figsize=(11, 8), sharex=True)
    for j, ax in enumerate(axes.flat):
        lin = j < 2
        y = q[:, j] if lin else np.degrees(q[:, j])
        ax.plot(t, y, lw=0.8, color="#1f4e79")
        ax.fill_between(t, y.min(), y.max(), where=spray, color="#f4a261", alpha=0.25, lw=0)
        ax.set_ylabel(ARM_JOINTS[j] + (" [m]" if lin else " [deg]"))
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle("Joint trajectories: gantry X, gantry Z + 6 robot axes (orange = spray gun ON)")
    fig.tight_layout()
    fig.savefig(PLOTS / "joint_trajectories.png", dpi=150)
    plt.close(fig)

    # 2. clearance
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, dmin * 1000, lw=0.8, color="#2a9d8f")
    ax.axhline(CONTACT_MARGIN * 1000, color="#e76f51", ls="--",
               label=f"planning safety margin {CONTACT_MARGIN*1000:.0f} mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("min clearance [mm]")
    ax.set_title("Minimum robot clearance to car, booth and itself (every sample)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "clearance.png", dpi=150)
    plt.close(fig)

    # 3. coverage bar
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = list(cov.keys())
    vals = [cov[n] * 100 for n in names]
    ax.bar(names, vals, color="#264653")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("coverage [%]")
    ax.set_title("Paint coverage per surface (film >= 50 % of nominal)")
    fig.tight_layout()
    fig.savefig(PLOTS / "coverage.png", dpi=150)
    plt.close(fig)

    # 3b. film thickness maps (unfolded surfaces)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    views = {"roof": (0, 1), "hood": (0, 1), "trunk": (0, 1), "front": (1, 2), "rear": (1, 2),
             "left_side": (0, 2), "right_side": (0, 2)}
    sc = None
    for ax, (name, (i, j)) in zip(axes.flat, views.items()):
        f = film[name]
        P = f["points"]
        sc = ax.scatter(P[:, i], P[:, j], c=f["rel"], s=4, cmap="viridis", vmin=0, vmax=1.3)
        ax.set_aspect("equal")
        ax.set_title(f"{name}  (CV {f['cv']:.2f})", fontsize=10)
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
    axes.flat[-1].axis("off")
    fig.colorbar(sc, ax=axes.flat[-1], fraction=0.8, label="film thickness / nominal")
    fig.suptitle("Simulated paint film thickness (Gaussian spray model)")
    fig.tight_layout()
    fig.savefig(PLOTS / "film_thickness.png", dpi=150)
    plt.close(fig)

    # 4. standoff
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(dots[dots[:, 8] > 0, 7] * 1000, bins=60, color="#457b9d")
    ax.axvline(paths.STANDOFF * 1000, color="#e63946", ls="--", label="target 250 mm")
    ax.set_xlabel("gun-to-surface distance while spraying [mm]")
    ax.set_ylabel("samples")
    ax.legend()
    ax.set_title("Standoff distance accuracy")
    fig.tight_layout()
    fig.savefig(PLOTS / "standoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
