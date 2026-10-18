"""Step 4: render the demo video (MP4) with camera moves and an on-screen HUD.

    python make_video.py                 # 1280x720, 30 fps, program at 4x speed
    python make_video.py --quick         # low-res preview (fast)
    python make_video.py --speed 3 --width 1920 --height 1080

Sequence:  title / cell orbit  ->  full painting program (car enters, 7 surfaces,
car leaves)  ->  collision-checking demo  ->  final orbit of the painted car.
Output: ./output/demo_video.mp4
"""
import argparse
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paintcell.playback import Playback, SmoothCamera, OUTPUT
from paintcell.render import highlight_collision

AUTHOR = "Vidya Sagar"
TITLE = "Robotic Car Painting Cell - MuJoCo Simulation"
SUBTITLE = "6-axis robot (IRB 6700 class) on a 2-axis overhead gantry  |  8 axes"


def load_font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else
             ["DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "DejaVuSans.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Hud:
    def __init__(self, w, h):
        s = h / 720
        self.f_title = load_font(int(30 * s), bold=True)
        self.f_big = load_font(int(22 * s), bold=True)
        self.f = load_font(int(17 * s))
        self.s = s

    def panel(self, img, lines, header=None):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        s = self.s
        x0, y0 = int(18 * s), int(18 * s)
        lh = int(24 * s)
        hh = int(32 * s) if header else 0
        w = int(520 * s)
        d.rounded_rectangle([x0, y0, x0 + w, y0 + hh + lh * len(lines) + int(14 * s)],
                            radius=int(10 * s), fill=(15, 20, 30, 170))
        y = y0 + int(8 * s)
        if header:
            d.text((x0 + int(14 * s), y), header, font=self.f_big, fill=(255, 200, 80))
            y += hh
        for ln in lines:
            col = (120, 255, 140) if "ON" in ln and "Spray" in ln else (235, 235, 235)
            d.text((x0 + int(14 * s), y), ln, font=self.f, fill=col)
            y += lh
        return np.asarray(im)

    def banner(self, img, title, sub=None, alpha=1.0):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        W, H = im.size
        bh = int(110 * self.s)
        y0 = H - bh - int(30 * self.s)
        d.rectangle([0, y0, W, y0 + bh], fill=(10, 15, 25, int(190 * alpha)))
        d.text((int(40 * self.s), y0 + int(18 * self.s)), title, font=self.f_title,
               fill=(255, 255, 255, int(255 * alpha)))
        if sub:
            d.text((int(40 * self.s), y0 + int(62 * self.s)), sub, font=self.f_big,
                   fill=(255, 200, 80, int(255 * alpha)))
        return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=4.0, help="program playback speed-up")
    ap.add_argument("--quick", action="store_true", help="640x360, 15 fps preview")
    ap.add_argument("--out", default=str(OUTPUT / "demo_video.mp4"))
    a = ap.parse_args()
    if a.quick:
        a.width, a.height, a.fps = 640, 360, 15

    pb = Playback()
    m, d = pb.cell.m, pb.cell.d
    r = mujoco.Renderer(m, a.height, a.width, max_geom=20000)
    hud = Hud(a.width, a.height)
    cam = mujoco.MjvCamera()
    scam = SmoothCamera(cam)
    dt = 1.0 / a.fps
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                                macro_block_size=8)
    nframes = 0
    t_start = time.time()

    def emit(frame):
        nonlocal nframes
        writer.append_data(frame)
        nframes += 1
        if nframes % 300 == 0:
            print(f"   {nframes} frames ({time.time() - t_start:.0f} s)")

    # ---- 1. title + orbit of the cell (car waiting at the entry) ----------
    T1 = 7.0
    pb.apply(0.0)
    for k in range(int(T1 * a.fps)):
        s = k / (T1 * a.fps)
        cam.lookat[:] = [-0.5, 0.5, 1.2]
        cam.azimuth, cam.elevation, cam.distance = 20 + 70 * s, -28, 15 - 3 * s
        r.update_scene(d, cam)
        img = r.render()
        emit(hud.banner(img, TITLE, f"{AUTHOR}   |   {SUBTITLE}", alpha=min(1, 2 * (1 - s) + 0.3)))

    # ---- 2. the painting program ------------------------------------------
    t = 0.0
    scam.init = False
    while t <= pb.duration:
        pb.apply(t)
        look, az, el, dist = pb.camera_target(t)
        scam.update(look, az, el, dist, dt)
        r.update_scene(d, cam)
        pb.decorate(r.scene, t)
        img = hud.panel(r.render(), pb.hud_lines(t, a.speed), header="Painting program")
        emit(img)
        t += dt * a.speed

    # ---- 3. collision-checking demo -----------------------------------------
    rej = OUTPUT / "rejected.npz"
    if rej.exists():
        z = np.load(rej)
        q0, q1, lab = z["q0"][0], z["q1"][0], str(z["labels"][0])
        t_lab = float(pb.t[np.isin(pb.label_idx, [i for i, l in enumerate(pb.labels)
                                                   if l.startswith(lab)])][0])
        base = (m.geom_rgba.copy(), m.geom_matid.copy())
        T3 = 8.0
        n = int(T3 * a.fps)
        hit_any = False
        for k in range(n):
            s = min(1.0, k / (0.8 * n))
            b = 10 * s**3 - 15 * s**4 + 6 * s**5
            pb.cell.set(q0 + b * (q1 - q0), 0.0)
            hit = highlight_collision(m, pb.cc, base)
            hit_any |= hit
            p, _ = pb.cell.tcp_pose()
            cam.lookat[:] = 0.5 * p + 0.5 * np.array([0, 0, 1.6])
            cam.azimuth, cam.elevation, cam.distance = 65 + 15 * k / n, -22, 8.0
            r.update_scene(d, cam)
            pb.paint.add_to_scene(r.scene, t_lab, 0.0)
            lines = [f"Checking: '{lab}' as a direct joint move",
                     "Contact check every sample (30 mm margin)",
                     ("COLLISION DETECTED -> motion rejected" if hit_any else "no contact yet ..."),
                     "Re-planned automatically (retract / RRT-Connect)"]
            emit(hud.panel(r.render(), lines, header="Collision checking demo"))
        m.geom_rgba[:], m.geom_matid[:] = base

    # ---- 4. final orbit of the painted car ------------------------------------
    T4 = 8.0
    t_done = float(pb.t[np.isin(pb.label_idx, [i for i, l in enumerate(pb.labels)
                                               if "return to park" in l])][-1])
    pb.apply(t_done)
    res = __import__("json").loads((OUTPUT / "results.json").read_text())
    for k in range(int(T4 * a.fps)):
        s = k / (T4 * a.fps)
        cam.lookat[:] = [0, 0, 0.9]
        cam.azimuth, cam.elevation, cam.distance = 100 + 200 * s, -25, 8.5
        r.update_scene(d, cam)
        pb.decorate(r.scene, t_done)
        img = r.render()
        img = hud.banner(img, "Result: all exterior surfaces painted",
                         f"coverage {res['total_coverage_pct']} %   |   0 collisions   |   "
                         f"min clearance {res['min_clearance_mm']:.0f} mm   |   "
                         f"cycle {res['cycle_time_s']:.0f} s")
        emit(img)

    writer.close()
    print(f"== wrote {a.out}: {nframes} frames, {nframes / a.fps:.0f} s of video "
          f"({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
