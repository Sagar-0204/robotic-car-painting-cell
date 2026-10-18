"""Builds the complete painting program (time-stamped joint trajectory).

Program structure
  1. Conveyor brings the car into the booth (robot parked).
  2. For every surface in paths.ORDER:
       a. choose a collision-free starting posture (multi-seed IK)
       b. joint-space transfer move to the first stroke  (quintic, collision-checked;
          if it collides a retract-and-transfer via the park posture is inserted)
       c. raster strokes: TCP moves at constant PAINT_SPEED, gun ON
       d. row changes: straight Cartesian TCP moves, gun OFF
  3. Robot parks, conveyor carries the car out.

Every sample of the final program is then collision checked and the spray is
ray-cast onto the car to measure coverage.
"""
import numpy as np

from . import ik, paths
from .cell import VEL_LIMITS, IDX
from .collision import CollisionChecker, SAFETY_MARGIN
from .rrt import RRTConnect

PAINT_SPEED = 0.40      # m/s TCP speed while spraying (typ. 0.3-0.6 m/s)
TRAVERSE_SPEED = 0.50   # m/s TCP speed for gun-off Cartesian moves
CONVEYOR_SPEED = 0.50   # m/s
CAR_IN_X, CAR_OUT_X = -6.5, 6.5
DT_MOVE = 0.04          # s, sampling of joint-space moves


class Segment:
    """A piece of the program: arrays sampled in time."""
    def __init__(self, kind, label, t, q, car_x, spray):
        self.kind, self.label = kind, label
        self.t, self.q, self.car_x, self.spray = t, q, car_x, spray


def quintic(q0, q1, dt=DT_MOVE, vel=VEL_LIMITS):
    """Minimum-jerk joint move; duration set by the slowest joint (peak vel = 1.875*avg)."""
    T = max(0.6, 1.875 * np.max(np.abs(q1 - q0) / vel))
    n = max(2, int(np.ceil(T / dt)) + 1)
    s = np.linspace(0, 1, n)
    b = 10 * s**3 - 15 * s**4 + 6 * s**5
    return np.linspace(0, T, n), q0 + np.outer(b, q1 - q0)


class Planner:
    def __init__(self, cell, seed=0, log=print):
        self.cell = cell
        self.cc = CollisionChecker(cell)
        self.rng = np.random.default_rng(seed)
        self.log = log
        self.segments = []
        self.events = []          # human-readable planning log (for the report)
        self.rejected = []        # colliding motions that were rejected

    # ---------------------------------------------------------------- helpers
    def collision_free(self, qs, car_x=0.0, stride=1):
        """Contact check along a joint path (30 mm margin).
        Returns (ok, index_of_first_hit, (geom_a, geom_b))."""
        m = self.cell.m
        idx = list(range(0, len(qs), stride))
        if idx[-1] != len(qs) - 1:
            idx.append(len(qs) - 1)
        for i in idx:
            self.cell.set(qs[i], car_x)
            if self.cc.in_collision():
                c = self.cell.d.contact[0]
                return False, i, (m.geom(c.geom1).name, m.geom(c.geom2).name)
        return True, -1, ("", "")

    def path_clearance(self, qs, car_x=0.0, stride=5):
        """Exact minimum clearance along a path (for choosing between candidates)."""
        best = (np.inf, ("", ""))
        for i in range(0, len(qs), stride):
            self.cell.set(qs[i], car_x)
            d, a, b = self.cc.clearance()
            if d < best[0]:
                best = (d, (self.cell.m.geom(a).name, self.cell.m.geom(b).name))
        return best

    def start_candidates(self, tgt, axis, q_cur, n_seeds=60, lift_pref=None):
        """Multi-start IK: converged, collision-free postures for the first stroke point,
        best first (clearance, joint-limit margin, distance from singularity, simplicity)."""
        cands = []
        seeds = [q_cur.copy()]
        home = self.cell.home(tgt[0])
        for off in (-1.6, -1.2, -0.6, 0.0, 0.6, 1.2, 1.6):            # deterministic seeds near park
            for j1 in (-np.pi / 2, 0.0, np.pi / 2, np.pi):
                q0 = home.copy()
                q0[0] = np.clip(tgt[0] + off, self.cell.lo[0], self.cell.hi[0])
                q0[IDX["j1"]] = j1
                for lift in (0.0, -0.7, -1.4):
                    q0 = q0.copy()
                    q0[IDX["lift"]] = lift
                    seeds.append(q0)
        for j1 in (-np.pi / 2, np.pi / 2, 0.0, np.pi):
            for j2 in (0.6, 1.0):
                for j5 in (-1.6, 1.6):
                    for lift in (-1.4, -0.7):
                        q0 = np.array([tgt[0], lift, j1, j2, -0.6, 0.0, j5, 0.0])
                        q0[0] = np.clip(q0[0], self.cell.lo[0], self.cell.hi[0])
                        seeds.append(q0)
        for _ in range(n_seeds):
            q0 = self.rng.uniform(self.cell.lo, self.cell.hi)
            q0[0] = np.clip(tgt[0] + self.rng.uniform(-1.5, 1.5), self.cell.lo[0], self.cell.hi[0])
            seeds.append(q0)
        for q0 in seeds:
            if lift_pref is not None:                    # solve with the lift locked
                q0 = q0.copy()
                q0[1] = lift_pref
                lo, hi = self.cell.lo.copy(), self.cell.hi.copy()
                self.cell.lo[1] = self.cell.hi[1] = lift_pref
                try:
                    q, ok, _, _ = ik.solve(self.cell, tgt, axis, q0)
                finally:
                    self.cell.lo, self.cell.hi = lo, hi
            else:
                q, ok, _, _ = ik.solve(self.cell, tgt, axis, q0)
            if not ok:
                continue
            self.cell.set(q)
            if self.cc.in_collision():
                continue
            d, _, _ = self.cc.clearance()
            # keep wrist away from singularity (|j5| small) and joints away from limits
            r = slice(IDX["j1"], None)                     # revolute joints only
            margin = np.min(np.minimum(q[r] - self.cell.lo[r], self.cell.hi[r] - q[r]))
            wrist = abs(np.sin(q[IDX["j5"]]))
            # prefer simple postures: wrist roll joints (j4, j6) near zero,
            # close to the park posture, so transfer moves stay short and smooth
            ref = self.cell.home(q[0], q[1])
            score = min(d, 0.3) + 1.0 * min(margin, 0.5) + 0.2 * wrist \
                - 0.05 * np.linalg.norm(q[r] - ref[r]) \
                - 0.3 * (abs(q[IDX["j4"]]) + abs(q[IDX["j6"]]))
            cands.append((score, q))
        if not cands:
            raise RuntimeError(f"no collision-free IK solution for target {tgt}")
        cands.sort(key=lambda c: -c[0])
        distinct = []                       # keep clearly different postures only
        for _, q in cands:
            if all(np.linalg.norm(q - p) > 0.3 for p in distinct):
                distinct.append(q)
        return distinct

    def track(self, poses, q_start, q_nom, bias, abort_mm=None):
        """IK along a list of (tcp, axis) poses with warm starts.
        If abort_mm is given, stop early (return None) once the error exceeds it."""
        q = q_start.copy()
        out, worst = [], 0.0
        for tgt, ax in poses:
            # redundancy resolution: the gantry carries the arm - rail follows the
            # target X, lift follows the target height - so the 6 robot joints can
            # keep the same arm shape over the whole surface
            qn = q_nom.copy()
            qn[0] = np.clip(tgt[0] + bias[0], self.cell.lo[0], self.cell.hi[0])
            qn[1] = np.clip(tgt[2] + bias[1], self.cell.lo[1], self.cell.hi[1])
            q, ok, pe, ae = ik.solve(self.cell, tgt, ax, q, q_nom=qn,
                                     iters=60 if abort_mm else 200)
            worst = max(worst, pe)
            if abort_mm is not None and (pe > abort_mm / 1000 or ae > 0.02):
                return None, worst
            out.append(q.copy())
        return np.array(out), worst

    @staticmethod
    def stroke_poses(stroke):
        return [(w[:3] + paths.STANDOFF * w[3:], -w[3:]) for w in stroke]

    @staticmethod
    def line_poses(p0, a0, p1, a1, step=paths.STEP):
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
        out = []
        for s in np.linspace(0, 1, n):
            a = (1 - s) * a0 + s * a1
            out.append(((1 - s) * p0 + s * p1, a / np.linalg.norm(a)))
        return out

    def timed(self, qs, poses_spacing, speed):
        """Time stamps for a Cartesian path: constant TCP speed, slowed where a
        joint would exceed its velocity limit."""
        t = [0.0]
        slowed = 0
        for i in range(1, len(qs)):
            dt_nom = poses_spacing[i - 1] / speed
            dt_lim = np.max(np.abs(qs[i] - qs[i - 1]) / VEL_LIMITS)
            if dt_lim > dt_nom:
                slowed += 1
            t.append(t[-1] + max(dt_nom, dt_lim, 1e-3))
        return np.array(t), slowed

    def add(self, seg):
        self.segments.append(seg)

    def retract(self, q, dist=0.35, lift=0.15):
        """Cartesian retract: pull the gun back along its axis (and up) from pose q."""
        self.cell.set(q)
        p, a = self.cell.tcp_pose()
        p1 = p - dist * a + np.array([0, 0, lift])
        lp = self.line_poses(p, a, p1, a)
        qs, _ = self.track(lp, q, q, (q[0] - p[0], q[1] - p[2]))
        sp = [np.linalg.norm(lp[i + 1][0] - lp[i][0]) for i in range(len(lp) - 1)]
        tt, _ = self.timed(qs, sp, TRAVERSE_SPEED)
        return tt, qs

    def transfer(self, q0, q1, label, car_x=0.0):
        """Collision-checked transfer between two postures.

        Strategy (first collision-free option wins):
          A. direct quintic joint move
          B. retract from q0 -> joint move -> approach to q1 (reverse retract)
          C. as B, but routed through the park posture
        """
        cx = lambda n: np.full(n, car_x)
        off = lambda n: np.zeros(n, bool)
        t, qs = quintic(q0, q1)
        ok, i, pair = self.collision_free(qs, car_x)
        if ok:
            self.add(Segment("transfer", label, t, qs, cx(len(t)), off(len(t))))
            return
        self.rejected.append(dict(label=label, q0=q0, q1=q1, pair=pair))

        t0, r0 = self.retract(q0)
        t1, r1 = self.retract(q1)
        r1, t1 = r1[::-1], t1[-1] - t1[::-1]
        ok0 = self.collision_free(r0, car_x)[0]
        ok1 = self.collision_free(r1, car_x)[0]
        options = {}
        for use0 in (True, False):
            for use1 in (True, False):
                if (use0 and not ok0) or (use1 and not ok1):
                    continue
                a = r0[-1] if use0 else q0
                b = r1[0] if use1 else q1
                tag = ("retract / " if use0 else "") + "{}" + (" / approach" if use1 else "")
                options[tag.format("direct")] = (use0, use1, [a, b])
                options[tag.format("park")] = (use0, use1, [a, self.cell.home(a[0]),
                                                            self.cell.home(b[0]), b])
        # cheapest first: fewest pieces
        for name, (use0, use1, via) in sorted(options.items(), key=lambda kv: len(kv[1][2])):
            pieces = [("rowchange", t0, r0)] if use0 else []
            good = True
            for a, b in zip(via[:-1], via[1:]):
                tq, qq = quintic(a, b)
                if not self.collision_free(qq, car_x)[0]:
                    good = False
                    break
                pieces.append(("transfer", tq, qq))
            if use1:
                pieces.append(("rowchange", t1, r1))
            if good:
                msg = (f"{label}: direct joint move collides ({pair[0]} vs {pair[1]}) "
                       f"-> rejected, re-planned as {name}")
                self.events.append(msg)
                self.log("   " + msg)
                for kind, tq, qq in pieces:
                    self.add(Segment(kind, f"{label} ({name})", tq, qq, cx(len(tq)), off(len(tq))))
                return
        # D. sampling-based planner (RRT-Connect) between the retract poses
        a = r0[-1] if ok0 else q0
        b = r1[0] if ok1 else q1
        path = None
        for attempt in range(3):                          # different random seeds
            path = RRTConnect(self.cell, self.cc, car_x,
                              seed=len(self.segments) + 101 * attempt).plan(a, b, max_iter=8000)
            if path is not None:
                break
        if path is None:                                  # two easier legs via the park posture
            park = self.cell.home(0.5 * (a[0] + b[0]))
            leg1 = RRTConnect(self.cell, self.cc, car_x, seed=7).plan(a, park, max_iter=10000)
            leg2 = RRTConnect(self.cell, self.cc, car_x, seed=8).plan(park, b, max_iter=10000)
            if leg1 is None or leg2 is None:
                raise RuntimeError(f"{label}: no collision-free transfer found")
            path = leg1 + leg2[1:]
        msg = (f"{label}: direct joint move collides ({pair[0]} vs {pair[1]}) "
               f"-> rejected, re-planned with RRT-Connect "
               f"({len(path) - 2} via points)")
        self.events.append(msg)
        self.log("   " + msg)
        if ok0:
            self.add(Segment("rowchange", f"{label} (retract)", t0, r0, cx(len(t0)), off(len(t0))))
        for qa, qb in zip(path[:-1], path[1:]):
            tq, qq = quintic(qa, qb)
            self.add(Segment("transfer", f"{label} (RRT-Connect)", tq, qq, cx(len(tq)), off(len(tq))))
        if ok1:
            self.add(Segment("rowchange", f"{label} (approach)", t1, r1, cx(len(t1)), off(len(t1))))

    def run_patch(self, name, strokes, q_start, abort_mm=None):
        """Raster strokes (gun on) linked by gun-off row changes, from q_start.
        Returns (segments, stats), or (None, None) if tracking failed (abort_mm)."""
        segs = []
        q = q_start
        bias = None
        self.cell.set(q_start)
        p0 = self.cell.tcp_pose()[0]
        bias = (q_start[0] - p0[0], q_start[1] - p0[2])
        n_slowed, worst_ik, length = 0, 0.0, 0.0
        prev_end = None
        for st in strokes:
            poses = self.stroke_poses(st)
            if prev_end is not None:
                lp = self.line_poses(prev_end[0], prev_end[1], poses[0][0], poses[0][1])
                qs, _ = self.track(lp, q, q_start, bias, abort_mm)
                if qs is None:
                    return None, None
                sp = [np.linalg.norm(lp[i + 1][0] - lp[i][0]) for i in range(len(lp) - 1)]
                tt, _ = self.timed(qs, sp, TRAVERSE_SPEED)
                segs.append(Segment("rowchange", f"{name} row change", tt, qs,
                                    np.zeros(len(tt)), np.zeros(len(tt), bool)))
                q = qs[-1]
            qs, w = self.track(poses, q, q_start, bias, abort_mm)
            if qs is None:
                return None, None
            worst_ik = max(worst_ik, w)
            sp = [np.linalg.norm(poses[i + 1][0] - poses[i][0]) for i in range(len(poses) - 1)]
            length += sum(sp)
            tt, sl = self.timed(qs, sp, PAINT_SPEED)
            n_slowed += sl
            segs.append(Segment("paint", f"painting {name}", tt, qs,
                                np.zeros(len(tt)), np.ones(len(tt), bool)))
            q = qs[-1]
            prev_end = poses[-1]
        stats = dict(strokes=len(strokes), path_length_m=round(length, 2),
                     max_ik_error_mm=round(worst_ik * 1000, 2), samples_slowed=int(n_slowed))
        return segs, stats

    # ------------------------------------------------------------- the program
    def build(self):
        cell = self.cell
        q = cell.home(0.0)
        # 1. car enters
        T = (0 - CAR_IN_X) / CONVEYOR_SPEED
        t = np.arange(0, T + 1e-9, DT_MOVE)
        cx = CAR_IN_X + (0 - CAR_IN_X) * (t / T)
        self.add(Segment("conveyor", "car enters booth", t, np.tile(q, (len(t), 1)), cx,
                         np.zeros(len(t), bool)))
        self.stats = {}
        patch_by_name = {p["name"]: p for p in paths.PATCHES}
        for name in paths.ORDER:
            patch = patch_by_name[name]
            cell.set(q)
            strokes = paths.strokes_for_patch(cell, patch)
            first = self.stroke_poses(strokes[0])[0]
            # Evaluate several start postures over the WHOLE surface: IK tracking alone
            # does not know about obstacles, so a posture that is fine at the first
            # point can drift into the car later.  Keep the one with most clearance.
            best = None
            # surfaces with height variation (sides, front, rear): the lift follows
            # the TCP height, so start low enough to leave stroke for the top rows
            z = np.concatenate([st[:, 2] for st in strokes]) + 0.0
            lift_pref = None
            if z.max() - z.min() > 0.1:
                lo, hi = cell.lo[1], cell.hi[1]
                lift_pref = float(np.clip(lo + 0.03 + (first[0][2] - z.min()), lo, hi))
            cands = self.start_candidates(first[0], first[1], q, lift_pref=lift_pref)
            for k, q_start in enumerate(cands[:80]):
                segs, stats = self.run_patch(name, strokes, q_start, abort_mm=2.0)
                if segs is None:
                    continue                        # cannot track the whole surface
                allq = np.vstack([sg.q for sg in segs])
                ok, _, pair = self.collision_free(allq, stride=2)
                dmin = self.path_clearance(allq)[0] if ok else -1.0
                # a posture must track the whole surface accurately (< 2 mm) AND be
                # collision free; among those prefer the largest clearance
                score = dmin - (1.0 if stats["max_ik_error_mm"] > 2.0 else 0.0)
                if best is None or score > best[0]:
                    best = (score, q_start, segs, stats, pair, dmin)
                if dmin > 0.10 and stats["max_ik_error_mm"] <= 2.0:
                    break
            if best is None:
                raise RuntimeError(f"{name}: no start posture can track the whole surface")
            score, q_start, segs, stats, pair, dmin = best
            if stats["max_ik_error_mm"] > 2.0:
                self.log(f"   WARNING {name}: best posture still has IK error "
                         f"{stats['max_ik_error_mm']} mm (reach/joint-limit issue)")
            if dmin < 0:
                raise RuntimeError(f"{name}: every candidate posture collides ({pair})")
            self.log(f"[{name}] {len(strokes)} strokes, rail={q_start[0]:+.2f} m "
                     f"lift={q_start[1]:+.2f} m, tested {k + 1}/{len(cands)} posture(s), "
                     f"min clearance {dmin*1000:.0f} mm")
            self.transfer(q, q_start, f"move to {name}")
            for sg in segs:
                self.add(sg)
            q = segs[-1].q[-1]
            self.stats[name] = stats
        # 3. park and car exits
        self.transfer(q, cell.home(0.0), "return to park")
        q = cell.home(0.0)
        T = CAR_OUT_X / CONVEYOR_SPEED
        t = np.arange(0, T + 1e-9, DT_MOVE)
        self.add(Segment("conveyor", "car leaves booth", t, np.tile(q, (len(t), 1)),
                         CAR_OUT_X * t / T, np.zeros(len(t), bool)))
        return self.flatten()

    def flatten(self):
        """Concatenate all segments into one program dict of arrays."""
        T, Q, CX, SP, LAB, KIND = [], [], [], [], [], []
        t0 = 0.0
        labels = []
        for s in self.segments:
            if labels and labels[-1] == s.label:
                li = len(labels) - 1
            else:
                labels.append(s.label)
                li = len(labels) - 1
            T.append(s.t + t0)
            Q.append(s.q)
            CX.append(s.car_x)
            SP.append(s.spray)
            LAB.append(np.full(len(s.t), li))
            KIND.append(np.full(len(s.t), ["conveyor", "transfer", "rowchange", "paint"].index(s.kind)))
            t0 = T[-1][-1] + 1e-3
        return dict(t=np.concatenate(T), q=np.vstack(Q), car_x=np.concatenate(CX),
                    spray=np.concatenate(SP), label_idx=np.concatenate(LAB),
                    kind=np.concatenate(KIND), labels=np.array(labels))
