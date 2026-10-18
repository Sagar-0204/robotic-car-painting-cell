"""Step 3: render high-resolution screenshots for the report.

    python take_screenshots.py          (run plan_program.py first)

Saves PNGs to ./output/screenshots/
"""
import mujoco
import numpy as np
from PIL import Image

from paintcell.playback import Playback, OUTPUT
from paintcell.render import highlight_collision

W, H = 1600, 900
OUT = OUTPUT / "screenshots"


def time_of(pb, label_part, frac=0.5, spray=None):
    """A time inside the first program part whose label contains label_part."""
    idx = [i for i, l in enumerate(pb.labels) if label_part in l]
    mask = np.isin(pb.label_idx, idx)
    if spray is not None:
        mask &= pb.spray == spray
    ts = pb.t[mask]
    return float(ts[int(frac * (len(ts) - 1))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pb = Playback()
    m, d = pb.cell.m, pb.cell.d
    r = mujoco.Renderer(m, H, W, max_geom=20000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()

    def shot(name, t, look, az, el, dist, collide=None):
        if collide is None:
            pb.apply(t)
        cam.lookat[:] = look
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(d, cam, opt)
        pb.decorate(r.scene, t)
        Image.fromarray(r.render()).save(OUT / f"{name}.png")
        print("  saved", name)

    T_end = pb.duration
    t_paint_done = time_of(pb, "return to park", 0.0)
    shot("01_cell_overview", 0.0, [0, 1.2, 1.2], 55, -32, 15.5)
    shot("02_top_layout", 0.0, [0, 1.5, 0], 90, -89.9, 15.0)
    t = time_of(pb, "painting roof", 0.6)
    shot("03_painting_roof", t, [-0.3, 0, 1.8], 55, -35, 7.0)
    t = time_of(pb, "painting hood", 0.6)
    shot("04_painting_hood", t, [1.6, 0, 1.6], 30, -30, 6.5)
    t = time_of(pb, "painting front", 0.5)
    shot("05_painting_front", t, [2.4, 0, 1.2], 155, -15, 6.0)
    t = time_of(pb, "painting right_side", 0.55)
    shot("06_painting_right_side", t, [0, -0.6, 1.4], 90, -12, 6.8)
    t = time_of(pb, "painting rear", 0.5)
    shot("07_painting_rear", t, [-2.4, 0, 1.2], 25, -15, 6.0)
    t = time_of(pb, "painting left_side", 0.5)
    shot("08_painting_left_side", t, [0, 0.6, 1.3], -70, -30, 6.5)
    t = time_of(pb, "painting trunk", 0.5)
    shot("09_painting_trunk", t, [-2.0, 0, 1.6], 140, -35, 6.5)
    shot("10_painted_car", t_paint_done, [0, 0, 1.0], 125, -28, 8.0)
    shot("11_car_exit", T_end, [4.5, 0, 1.0], 60, -25, 9.0)

    # collision check demo: the first rejected direct move, at its worst point
    rej = OUTPUT / "rejected.npz"
    if rej.exists():
        z = np.load(rej)
        q0, q1 = z["q0"][0], z["q1"][0]
        base = (m.geom_rgba.copy(), m.geom_matid.copy())
        best = None
        for s in np.linspace(0, 1, 60):
            pb.cell.set(q0 + s * (q1 - q0), 0.0)
            n = len(pb.cc.contacts())
            if best is None or n > best[0]:
                best = (n, s)
        q = q0 + best[1] * (q1 - q0)
        pb.cell.set(q, 0.0)
        highlight_collision(m, pb.cc, base)
        p, _ = pb.cell.tcp_pose()
        tlab = time_of(pb, str(z["labels"][0]), 0.0)
        cam.lookat[:] = 0.5 * p + 0.5 * np.array([0, 0, 1.6])
        cam.azimuth, cam.elevation, cam.distance = 65, -22, 8.0
        r.update_scene(d, cam, opt)
        pb.paint.add_to_scene(r.scene, tlab, 0.0)
        Image.fromarray(r.render()).save(OUT / "12_collision_detected.png")
        m.geom_rgba[:], m.geom_matid[:] = base
        print("  saved 12_collision_detected  (", str(z["labels"][0]), ")")


if __name__ == "__main__":
    main()
