"""Collision checking for the painting robot.

Two complementary checks:
  1. Contact check (fast, used inside the planners)
     MuJoCo's collision detector (mj_collision) with a 30 mm margin on every
     robot geom.  Robot links are set to collide with the car, the booth walls,
     the gantry columns and the floor (contype/conaffinity bits), plus explicit
     <pair>s for self-collision between non-adjacent links.  Any contact means
     "closer than 30 mm" -> the configuration is rejected.
  2. Clearance (exact, used for verification/reporting)
     Signed distance between every robot link and every obstacle with
     mj_geomDistance, giving a continuous safety margin over the whole program.
"""
import mujoco
import numpy as np

SAFETY_MARGIN = 0.05          # m, reporting threshold
CONTACT_MARGIN = 0.03         # m, set as geom margin in the MJCF
DIST_MAX = 1.0                # m, distances above this are not computed exactly


class CollisionChecker:
    def __init__(self, cell):
        self.cell = cell
        m = cell.m
        obstacles = cell.car_geoms + cell.env_geoms
        pairs = [(r, o) for r in cell.robot_geoms for o in obstacles]
        pairs += [(m.pair_geom1[i], m.pair_geom2[i]) for i in range(m.npair)]
        self.pairs = np.array(pairs)
        self.robot_set = set(cell.robot_geoms)
        self.plane = np.array([m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in range(m.ngeom)])

    # ---------------------------------------------------------------- fast
    def in_collision(self):
        """True if any robot link is within CONTACT_MARGIN of anything.
        Call after cell.set(...)."""
        mujoco.mj_collision(self.cell.m, self.cell.d)
        return self.cell.d.ncon > 0

    def contacts(self):
        mujoco.mj_collision(self.cell.m, self.cell.d)
        d = self.cell.d
        return [(d.contact[i].geom1, d.contact[i].geom2, d.contact[i].dist) for i in range(d.ncon)]

    # ---------------------------------------------------------------- exact
    def clearance(self):
        """Minimum signed distance (robot<->world, robot<->robot).
        Returns (dmin, geom_a, geom_b).  A bounding-sphere broad phase skips far pairs."""
        m, d = self.cell.m, self.cell.d
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        centre = np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b], axis=1)
        approx = centre - m.geom_rbound[a] - m.geom_rbound[b]
        approx[self.plane[a] | self.plane[b]] = -np.inf          # planes: always check
        best = (DIST_MAX, -1, -1)
        for k in np.where(approx < DIST_MAX)[0]:
            dist = mujoco.mj_geomDistance(m, d, int(a[k]), int(b[k]), DIST_MAX, None)
            if dist < best[0]:
                best = (dist, int(a[k]), int(b[k]))
        return best

    def check(self, q, car_x=0.0):
        """Full check of one configuration -> (dmin, name_a, name_b, n_contacts)."""
        self.cell.set(q, car_x)
        dmin, a, b = self.clearance()
        n = len(self.contacts())
        m = self.cell.m
        return (dmin, m.geom(a).name if a >= 0 else "", m.geom(b).name if b >= 0 else "", n)
