"""Painting path generation by ray-casting the car surface.

For each surface ("patch") we cast a grid of parallel rays at the car from
outside.  Every ray that hits a paintable panel gives a surface point and its
outward normal n.  The spray gun target for that point is

        TCP = hit + STANDOFF * n        spray axis = -n

Points are organised in rows (raster / zig-zag pattern): along a row the gun
moves at constant speed, rows are ROW_PITCH apart so that consecutive spray
bands overlap by ~50 %.  Rays that hit glass or miss the car split a row into
separate strokes (gun is switched off over openings).
"""
import mujoco
import numpy as np

STANDOFF = 0.25        # m, gun-to-surface distance (typical 200-300 mm)
FAN_WIDTH = 0.30       # m, spray pattern width on the surface at that standoff
ROW_PITCH = 0.15       # m, 50 % overlap between passes
STEP = 0.025           # m, waypoint spacing along a stroke
OVERSPRAY = 0.10       # m, gun stays on this far past each panel edge (uniform edges)

# ---------------------------------------------------------------- surfaces
# dir   : ray direction (towards the car)
# origin: a point on the ray start plane (outside the car)
# u, v  : stroke direction and row-step direction inside that plane
# urange / vrange : extents of the scan in the plane (car frame, metres)
# normal: required outward normal of the painted face
# geoms : panels that belong to this surface
PATCHES = [
    dict(name="roof",  dir=(0, 0, -1), origin=(0, 0, 3.0), u=(1, 0, 0), v=(0, 1, 0),
         urange=(-1.40, 0.90), vrange=(-0.72, 0.72), normal=(0, 0, 1), geoms=["cabin"]),
    dict(name="hood",  dir=(0, 0, -1), origin=(0, 0, 3.0), u=(0, 1, 0), v=(1, 0, 0),
         urange=(-0.92, 0.92), vrange=(1.55, 2.24), normal=(0, 0, 1), geoms=["lower_body"]),
    dict(name="trunk", dir=(0, 0, -1), origin=(0, 0, 3.0), u=(0, 1, 0), v=(1, 0, 0),
         urange=(-0.92, 0.92), vrange=(-2.24, -1.94), normal=(0, 0, 1), geoms=["lower_body"]),
    dict(name="left_side",  dir=(0, -1, 0), origin=(0, 3.0, 0), u=(1, 0, 0), v=(0, 0, 1),
         urange=(-2.40, 2.40), vrange=(0.40, 1.55), normal=(0, 1, 0),
         geoms=["lower_body", "cabin", "bumper_front", "bumper_rear"]),
    dict(name="right_side", dir=(0, 1, 0), origin=(0, -3.0, 0), u=(1, 0, 0), v=(0, 0, 1),
         urange=(-2.40, 2.40), vrange=(0.40, 1.55), normal=(0, -1, 0),
         geoms=["lower_body", "cabin", "bumper_front", "bumper_rear"]),
    dict(name="front", dir=(-1, 0, 0), origin=(3.5, 0, 0), u=(0, 1, 0), v=(0, 0, 1),
         urange=(-0.88, 0.88), vrange=(0.36, 0.98), normal=(1, 0, 0),
         geoms=["bumper_front", "lower_body"]),
    dict(name="rear",  dir=(1, 0, 0), origin=(-3.5, 0, 0), u=(0, 1, 0), v=(0, 0, 1),
         urange=(-0.88, 0.88), vrange=(0.36, 0.98), normal=(-1, 0, 0),
         geoms=["bumper_rear", "lower_body"]),
]
# order in which the robot paints them (top surfaces first, then around the car)
ORDER = ["roof", "hood", "front", "right_side", "rear", "left_side", "trunk"]


def box_normal(m, d, gid, p):
    """Outward normal of a box geom at surface point p."""
    R = d.geom_xmat[gid].reshape(3, 3)
    local = R.T @ (p - d.geom_xpos[gid])
    k = np.argmax(np.abs(local) / m.geom_size[gid])
    n = np.zeros(3)
    n[k] = np.sign(local[k])
    return R @ n


def cast(cell, origin, direction):
    """Cast one ray against car panels + glass (groups 2-3). Returns (geom_id, point) or (-1, None)."""
    m, d = cell.m, cell.d
    geomgroup = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.asarray(origin, float), np.asarray(direction, float),
                         geomgroup, 1, -1, gid)
    if gid[0] < 0 or dist < 0:
        return -1, None
    return int(gid[0]), np.asarray(origin) + dist * np.asarray(direction)


def scan_patch(cell, patch, step_u=STEP, pitch=ROW_PITCH, extend=0.0):
    """Ray-cast a patch.  Returns a list of rows; each row is a list of
    (u, point, normal) for valid hits, or None for a miss / glass / wrong face."""
    m, d = cell.m, cell.d
    o = np.array(patch["origin"], float)
    dr = np.array(patch["dir"], float)
    u, v = np.array(patch["u"], float), np.array(patch["v"], float)
    want_n = np.array(patch["normal"], float)
    allowed = {m.geom(g).id for g in patch["geoms"]}
    u0, u1 = patch["urange"][0] - extend, patch["urange"][1] + extend
    v0, v1 = patch["vrange"][0] - extend, patch["vrange"][1] + extend
    nrow = int(round((v1 - v0) / pitch)) + 1
    vs = np.linspace(v0, v1, max(nrow, 1))
    us = np.arange(u0, u1 + 1e-9, step_u)
    rows = []
    for vv in vs:
        row = []
        for uu in us:
            start = o + uu * u + vv * v
            gid, p = cast(cell, start, dr)
            if gid in allowed:
                n = box_normal(m, d, gid, p)
                if np.dot(n, want_n) > 0.99:
                    row.append((uu, p, n))
                    continue
            row.append(None)
        rows.append((vv, row))
    return rows


def strokes_for_patch(cell, patch):
    """Turn the scan into an ordered list of strokes (zig-zag).
    Each stroke is an array of waypoints: rows [x y z nx ny nz] (surface point + normal)."""
    rows = scan_patch(cell, patch)
    strokes = []
    for i, (vv, row) in enumerate(rows):
        segs, cur = [], []
        for item in row:
            if item is None:
                if len(cur) >= 3:
                    segs.append(cur)
                cur = []
            else:
                # split if the surface depth jumps (e.g. body side -> cabin side)
                if cur and np.linalg.norm(item[1] - cur[-1][1]) > 2.5 * STEP:
                    if len(cur) >= 3:
                        segs.append(cur)
                    cur = []
                cur.append(item)
        if len(cur) >= 3:
            segs.append(cur)
        if i % 2 == 1:                     # zig-zag: reverse every second row
            segs = [s[::-1] for s in segs[::-1]]
        for s in segs:
            w = np.array([np.concatenate([p, n]) for _, p, n in s])
            strokes.append(extend_stroke(w))
    return strokes


def extend_stroke(w, dist=OVERSPRAY):
    """Extend a stroke past both ends along its own direction (same depth/normal)
    so the spray fan fully covers the panel edge."""
    k = int(round(dist / STEP))
    d0 = w[0, :3] - w[1, :3]
    d1 = w[-1, :3] - w[-2, :3]
    d0, d1 = d0 / np.linalg.norm(d0) * STEP, d1 / np.linalg.norm(d1) * STEP
    pre = [np.concatenate([w[0, :3] + d0 * i, w[0, 3:]]) for i in range(k, 0, -1)]
    post = [np.concatenate([w[-1, :3] + d1 * i, w[-1, 3:]]) for i in range(1, k + 1)]
    return np.vstack(pre + [w] + post)


def surface_samples(cell, patch, res=0.02):
    """Dense samples of the surface a patch is responsible for (for coverage)."""
    pts = []
    for _, row in scan_patch(cell, patch, step_u=res, pitch=res, extend=0.1):
        for item in row:
            if item is not None:
                pts.append(np.concatenate([item[1], item[2]]))
    return np.array(pts)
