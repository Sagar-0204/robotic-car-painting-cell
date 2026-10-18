"""Model loading and small kinematic helpers for the paint cell."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "paint_cell.xml"

ARM_JOINTS = ["rail", "lift", "j1", "j2", "j3", "j4", "j5", "j6"]   # 8 axes: gantry X, Z + 6R
IDX = {n: i for i, n in enumerate(ARM_JOINTS)}

# Velocity limits used for timing and checking (rail/lift in m/s, joints in rad/s).
# Based on IRB 6700 class data sheet values, reduced ~40 % for a paint process.
VEL_LIMITS = np.array([1.0, 0.5] + list(np.radians([60, 60, 60, 100, 100, 140])))

PAINTABLE = ["lower_body", "cabin", "bumper_front", "bumper_rear"]
GLASS = ["windshield", "rear_window", "window_l", "window_r"]


class Cell:
    """Wraps the MuJoCo model + data and exposes robot-level helpers."""

    def __init__(self, xml_path=MODEL_PATH):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.dadr = np.array([m.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.lo = np.array([m.joint(n).range[0] for n in ARM_JOINTS])
        self.hi = np.array([m.joint(n).range[1] for n in ARM_JOINTS])
        self.conv_adr = m.joint("conveyor").qposadr[0]
        self.tcp = m.site("tcp").id
        self.car_body = m.body("car").id
        self.paint_geoms = {m.geom(n).id for n in PAINTABLE}
        self.glass_geoms = {m.geom(n).id for n in GLASS}

        # robot geoms = every geom in the carriage subtree that can collide
        carriage = m.body("carriage").id
        def in_robot(b):
            while b != 0:
                if b == carriage:
                    return True
                b = m.body_parentid[b]
            return False
        self.robot_geoms = [g for g in range(m.ngeom)
                            if in_robot(m.geom_bodyid[g]) and m.geom_contype[g] == 1]
        self.car_geoms = [g for g in range(m.ngeom)
                          if m.geom_bodyid[g] == self.car_body and m.geom_conaffinity[g] == 1]
        self.env_geoms = [g for g in range(m.ngeom)
                          if m.geom_bodyid[g] == 0 and m.geom_conaffinity[g] == 1]
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

    # ------------------------------------------------------------------ state
    def set(self, q, car_x=0.0, full=False):
        """Write the 7 arm coordinates (+ conveyor) and update kinematics.
        full=True also runs collision detection (mj_forward)."""
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.conv_adr] = car_x
        if full:
            mujoco.mj_forward(self.m, self.d)
        else:
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)

    def tcp_pose(self):
        """TCP position and spray axis (site local +X) in world frame."""
        R = self.d.site_xmat[self.tcp].reshape(3, 3)
        return self.d.site_xpos[self.tcp].copy(), R[:, 0].copy()

    def jac(self):
        mujoco.mj_jacSite(self.m, self.d, self._jacp, self._jacr, self.tcp)
        return self._jacp[:, self.dadr], self._jacr[:, self.dadr]

    def home(self, rail=0.0, lift=0.0):
        """Folded 'park' posture: upper arm swung back, forearm folded forward under
        the gantry beam, gun horizontal.  Everything stays above z = 2.1 m so the car
        (1.6 m high) can pass underneath on the conveyor."""
        rail = float(np.clip(rail, -2.0, 1.8))      # folded arm must stay inside the booth
        return np.array([rail, lift, 0.0, np.radians(-80), np.radians(75), 0.0, 0.0, 0.0])
