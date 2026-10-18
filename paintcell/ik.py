"""Numerical inverse kinematics for the 7-axis (rail + 6R) painting robot.

Task: put the TCP at position p_t with the spray axis along a_t.
A spray bell is rotationally symmetric, so rotation ABOUT the spray axis is
free -> the task is 5-D.  With 7 joints the robot has 2 redundant DOF, which
we use (null space) to keep the arm close to a preferred posture and the
carriage at a preferred offset from the target.

Method: damped least squares (Levenberg-Marquardt style)
    dq = J^T (J J^T + lambda^2 I)^-1 e  +  N (q_nom - q) * k
with N = I - J^+ J the null-space projector.
"""
import numpy as np


def _orientation_error(a, a_t):
    """Rotation vector that turns axis a onto a_t (small-angle form)."""
    e = np.cross(a, a_t)
    if np.dot(a, a_t) < 0.0:                       # more than 90 deg away
        n = np.linalg.norm(e)
        if n < 1e-6:                               # anti-parallel: pick any perpendicular
            e = np.cross(a, [1.0, 0.0, 0.0])
            if np.linalg.norm(e) < 1e-6:
                e = np.cross(a, [0.0, 1.0, 0.0])
            n = np.linalg.norm(e)
        e = e / n * 1.5                            # push hard toward the target
    return e


def solve(cell, p_t, a_t, q0, q_nom=None, k_null=0.1, car_x=0.0,
          iters=200, pos_tol=5e-4, ang_tol=2e-3, damping=0.05, max_step=0.25):
    """Returns (q, converged, pos_err, ang_err_rad)."""
    q = np.array(q0, dtype=float)
    a_t = np.asarray(a_t, float) / np.linalg.norm(a_t)
    p_t = np.asarray(p_t, float)
    I6 = np.eye(6)
    for _ in range(iters):
        cell.set(q, car_x)
        p, a = cell.tcp_pose()
        ep = p_t - p
        eo = _orientation_error(a, a_t)
        pe, ae = np.linalg.norm(ep), np.arccos(np.clip(np.dot(a, a_t), -1, 1))
        if pe < pos_tol and ae < ang_tol:
            return q, True, pe, ae
        jp, jr = cell.jac()
        P = np.eye(3) - np.outer(a, a)            # drop roll about the spray axis
        J = np.vstack([jp, P @ jr])
        e = np.concatenate([ep, P @ eo])
        JJt = J @ J.T + damping ** 2 * I6
        Jpinv = J.T @ np.linalg.inv(JJt)
        dq = Jpinv @ e
        if q_nom is not None:
            # exact projector (SVD pseudo-inverse) so the posture term never
            # disturbs the painting task to first order
            N = np.eye(len(q)) - np.linalg.pinv(J, rcond=1e-3) @ J
            dn = N @ (k_null * (q_nom - q))
            nn = np.linalg.norm(dn)
            if nn > 0.05:
                dn *= 0.05 / nn
            dq += dn
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q = np.clip(q + dq, cell.lo, cell.hi)
    cell.set(q, car_x)
    p, a = cell.tcp_pose()
    pe = np.linalg.norm(p_t - p)
    ae = np.arccos(np.clip(np.dot(a, a_t), -1, 1))
    return q, (pe < 1e-3 and ae < 1e-2), pe, ae
