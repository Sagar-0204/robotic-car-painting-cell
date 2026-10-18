"""RRT-Connect joint-space motion planner (fallback for transfer moves).

Used only when the simple transfer strategies (direct joint move, retract /
park detours) are blocked by an obstacle.  Classic algorithm (Kuffner & LaValle
2000): two trees grown from start and goal toward random samples, trying to
connect to each other; the found path is then shortened by random shortcuts.
"""
import numpy as np


class RRTConnect:
    def __init__(self, cell, checker, car_x=0.0, step=0.15, resolution=0.04,
                 rail_window=1.5, seed=0):
        self.cell, self.cc, self.car_x = cell, checker, car_x
        self.step, self.res = step, resolution
        self.rng = np.random.default_rng(seed)
        self.rail_window = rail_window
        # weight: 1 m of rail travel ~ 1 rad of joint motion
        self.lo, self.hi = cell.lo.copy(), cell.hi.copy()

    def free(self, q):
        self.cell.set(q, self.car_x)
        return not self.cc.in_collision()

    def edge_free(self, a, b):
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / self.res)) + 1)
        return all(self.free(a + s * (b - a)) for s in np.linspace(0, 1, n)[1:])

    def sample(self, q0, q1):
        q = self.rng.uniform(self.lo, self.hi)
        lo = min(q0[0], q1[0]) - self.rail_window
        hi = max(q0[0], q1[0]) + self.rail_window
        q[0] = self.rng.uniform(max(lo, self.lo[0]), min(hi, self.hi[0]))
        # keep the wrist roll joints near the range actually used
        for j in (5, 7):            # j4, j6 (wrist roll axes)
            q[j] = self.rng.uniform(min(q0[j], q1[j]) - 1.0, max(q0[j], q1[j]) + 1.0)
        return q

    def extend(self, tree, parent, q):
        d = np.linalg.norm(np.array(tree) - q, axis=1)
        i = int(np.argmin(d))
        near = tree[i]
        diff = q - near
        n = np.linalg.norm(diff)
        new = q if n <= self.step else near + diff / n * self.step
        if self.edge_free(near, new):
            tree.append(new)
            parent.append(i)
            return len(tree) - 1, np.allclose(new, q)
        return -1, False

    def plan(self, q0, q1, max_iter=4000):
        if not (self.free(q0) and self.free(q1)):
            return None
        Ta, Pa, Tb, Pb = [q0.copy()], [-1], [q1.copy()], [-1]
        swapped = False
        for _ in range(max_iter):
            qr = self.sample(q0, q1)
            ia, _ = self.extend(Ta, Pa, qr)
            if ia >= 0:
                # greedily connect tree B to the new node
                target = Ta[ia]
                while True:
                    ib, reached = self.extend(Tb, Pb, target)
                    if ib < 0:
                        break
                    if reached:
                        pa = self._trace(Ta, Pa, ia)
                        pb = self._trace(Tb, Pb, ib)
                        path = pa[::-1] + pb[1:]
                        if swapped:
                            path = path[::-1]
                        return self.shortcut(path)
            Ta, Pa, Tb, Pb = Tb, Pb, Ta, Pa
            swapped = not swapped
        return None

    @staticmethod
    def _trace(T, P, i):
        out = []
        while i >= 0:
            out.append(T[i])
            i = P[i]
        return out

    def shortcut(self, path, tries=150):
        path = list(path)
        for _ in range(tries):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.edge_free(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path
