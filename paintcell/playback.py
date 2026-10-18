"""Replays a saved painting program: robot/car state at any time t, paint layer,
camera choreography and on-screen text.  Used by the viewer, the video renderer
and the screenshot script."""
from pathlib import Path

import numpy as np

from . import paths
from .cell import Cell
from .collision import CollisionChecker
from .render import PaintLayer, add_spray

OUTPUT = Path(__file__).resolve().parent.parent / "output"

# camera presets per operation: (azimuth, elevation, distance)
#   azimuth 90 = looking along +Y (camera on the -Y side of the booth)
VIEWS = {
    "roof":       (60, -42, 7.5),
    "hood":       (35, -38, 7.0),
    "trunk":      (140, -38, 7.0),
    "front":      (160, -18, 6.0),
    "rear":       (20, -18, 6.0),
    "right_side": (90, -14, 6.5),
    "left_side":  (-70, -30, 6.5),
    "conveyor":   (70, -24, 10.0),
    "park":       (70, -24, 9.0),
}
PRETTY = {"roof": "roof", "hood": "hood", "trunk": "trunk lid", "front": "front face",
          "rear": "rear face", "right_side": "right side", "left_side": "left side"}


class Playback:
    def __init__(self, program=OUTPUT / "program.npz"):
        z = np.load(program, allow_pickle=False)
        self.t = z["t"]
        self.q = z["q"]
        self.car_x = z["car_x"]
        self.spray = z["spray"]
        self.label_idx = z["label_idx"]
        self.labels = [str(s) for s in z["labels"]]
        self.clearance = z["clearance"]
        self.dots = z["dots"]
        self.duration = float(self.t[-1])
        self.cell = Cell()
        self.cc = CollisionChecker(self.cell)
        self.paint = PaintLayer(self.dots)
        self.n_spray_total = int(self.spray.sum())
        self.spray_cum = np.cumsum(self.spray)

    # ------------------------------------------------------------- state at t
    def index(self, t):
        return int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 1))

    def state(self, t):
        """Interpolated joint vector and conveyor position at time t."""
        q = np.array([np.interp(t, self.t, self.q[:, j]) for j in range(self.q.shape[1])])
        return q, float(np.interp(t, self.t, self.car_x))

    def apply(self, t):
        q, cx = self.state(t)
        self.cell.set(q, cx)
        return q, cx

    def label(self, t):
        return self.labels[self.label_idx[self.index(t)]]

    def surface(self, t):
        lab = self.label(t)
        if "car" in lab:
            return "conveyor"
        if "park" in lab and "move to" not in lab:
            return "park"
        for name in paths.ORDER:
            if name in lab:
                return name
        return "park"

    def spraying(self, t):
        return bool(self.spray[self.index(t)])

    # ------------------------------------------------------------- drawing
    def decorate(self, scn, t):
        """Add paint and spray plume for time t to a scene (after update_scene)."""
        _, cx = self.state(t)
        self.paint.add_to_scene(scn, t, cx)
        if self.spraying(t):
            p, a = self.cell.tcp_pose()
            add_spray(scn, p, a)

    def hud_lines(self, t, speed=None):
        i = self.index(t)
        lab = self.label(t)
        pretty = lab
        for k, v in PRETTY.items():
            pretty = pretty.replace(k, v)
        done = self.spray_cum[i] / max(self.n_spray_total, 1)
        lines = [
            f"Operation : {pretty}",
            f"Spray gun : {'ON' if self.spray[i] else 'off'}",
            f"Painted   : {done*100:5.1f} % of program",
            f"Clearance : {self.clearance[i]*1000:5.0f} mm (min to car/booth/self)",
            f"Time      : {t:6.1f} s / {self.duration:.0f} s" + (f"   ({speed:g}x)" if speed else ""),
        ]
        return lines

    def camera_target(self, t):
        """(lookat, azimuth, elevation, distance) the camera should aim for."""
        s = self.surface(t)
        az, el, dist = VIEWS[s]
        _, cx = self.state(t)
        if s in ("conveyor", "park"):
            look = np.array([cx * 0.7, 0, 1.4])
        else:
            p, a = self.cell.tcp_pose()
            look = 0.55 * (p + 0.25 * a) + 0.45 * np.array([cx, 0, 1.0])
        return look, az, el, dist


class SmoothCamera:
    """First-order smoothing of the camera so cuts become glides."""

    def __init__(self, cam, tau=0.8):
        self.cam, self.tau = cam, tau
        self.init = False

    def update(self, look, az, el, dist, dt):
        c = self.cam
        if not self.init:
            c.lookat[:] = look
            c.azimuth, c.elevation, c.distance = az, el, dist
            self.init = True
            return
        k = 1 - np.exp(-dt / self.tau)
        c.lookat[:] = c.lookat + k * (look - c.lookat)
        daz = (az - c.azimuth + 180) % 360 - 180            # shortest way round
        c.azimuth += k * daz
        c.elevation += k * (el - c.elevation)
        c.distance += k * (dist - c.distance)
