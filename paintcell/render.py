"""Drawing helpers shared by the interactive viewer and the video renderer:
paint deposited on the car, the spray plume, and red highlighting of colliding links."""
import mujoco
import numpy as np

PAINT_RGBA = np.array([0.06, 0.22, 0.62, 1.0], dtype=np.float32)    # metallic blue
SPRAY_RGBA = np.array([0.55, 0.7, 1.0, 0.25], dtype=np.float32)
DOT_EVERY = 2               # draw every 2nd spray sample (5 cm apart) to save geoms


def _frame_from_normal(n):
    """3x3 rotation whose z-axis is n."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.column_stack([x, y, n])


class PaintLayer:
    """Paint shown as thin rectangular tiles (car frame): one tile per drawn spray
    sample, aligned with the stroke direction, as wide as the row pitch, so
    neighbouring strokes tile into a continuous coat."""

    def __init__(self, dots):
        from .paths import ROW_PITCH, STEP
        on = dots[:, 8] > 0 if dots.shape[1] > 8 else np.ones(len(dots), bool)
        d = dots[on][::DOT_EVERY]
        self.t = d[:, 0]
        self.pos = d[:, 1:4] + d[:, 4:7] * 0.003           # lift slightly off the panel
        P = d[:, 1:4]
        mats = []
        for i in range(len(d)):
            n = d[i, 4:7]
            # stroke direction from neighbouring samples of the same stroke
            dirs = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(d) and np.linalg.norm(P[j] - P[i]) < 3 * STEP * DOT_EVERY \
                        and np.dot(d[j, 4:7], n) > 0.99:
                    dirs.append((P[j] - P[i]) * (1 if j > i else -1))
            u = np.sum(dirs, axis=0) if dirs else np.cross(n, [0, 0, 1.0])
            u = u - np.dot(u, n) * n
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(n, [1.0, 0, 0])
            u /= np.linalg.norm(u)
            mats.append(np.column_stack([u, np.cross(n, u), n]).ravel())
        self.mat = np.array(mats)
        self.size = np.array([STEP * DOT_EVERY / 2 + 0.004, ROW_PITCH / 2 + 0.004, 0.0015])

    def count(self, t):
        return int(np.searchsorted(self.t, t, side="right"))

    def add_to_scene(self, scn, t, car_x, start=0):
        """Append paint tiles deposited up to time t. Returns number added."""
        n = self.count(t)
        size = self.size
        shift = np.array([car_x, 0, 0])
        added = 0
        for i in range(n):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, size,
                                self.pos[i] + shift, self.mat[i], PAINT_RGBA)
            scn.ngeom += 1
            added += 1
        return added


def add_spray(scn, tcp, axis, length=0.25):
    """Translucent spray plume from the nozzle to the surface."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    centre = tcp + axis * length / 2
    R = _frame_from_normal(axis)
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([0.07, 0.07, length / 2]), centre, R.ravel(), SPRAY_RGBA)
    scn.ngeom += 1


def highlight_collision(model, cc, base):
    """Colour robot geoms that are in contact (closer than the 30 mm margin) red.
    base = (geom_rgba copy, geom_matid copy) to restore from.  Requires cell.set()."""
    model.geom_rgba[:] = base[0]
    model.geom_matid[:] = base[1]
    hit = False
    for g1, g2, _ in cc.contacts():
        for g in (g1, g2):
            if g in cc.robot_set:
                model.geom_matid[g] = -1          # material colour would override rgba
                model.geom_rgba[g] = [1.0, 0.05, 0.05, 1.0]
                hit = True
    return hit
