"""Step 2: watch the painting program live in the interactive MuJoCo viewer.

    python view_simulation.py              (run plan_program.py first)
    python view_simulation.py --speed 4    start at 4x speed

Keys (click the viewer window first):
    SPACE   pause / resume
    ]  [    faster / slower  (x2 / /2)
    R       restart from the beginning
    C       toggle collision highlighting (robot links turn red within 30 mm)
Mouse: left-drag rotate, right-drag pan, scroll zoom (standard MuJoCo viewer).
"""
import argparse
import time

import mujoco
import mujoco.viewer

from paintcell.playback import Playback
from paintcell.render import highlight_collision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=2.0)
    a = ap.parse_args()

    pb = Playback()
    m, d = pb.cell.m, pb.cell.d
    base = (m.geom_rgba.copy(), m.geom_matid.copy())
    st = dict(t=0.0, speed=a.speed, paused=False, collide=True)

    def on_key(key):
        if key == 32:
            st["paused"] = not st["paused"]
        elif key == 93:
            st["speed"] = min(st["speed"] * 2, 64)
        elif key == 91:
            st["speed"] = max(st["speed"] / 2, 0.25)
        elif key == 82:
            st["t"] = 0.0
        elif key == 67:
            st["collide"] = not st["collide"]
        print(f"t={st['t']:.1f}s speed={st['speed']}x paused={st['paused']} "
              f"collision-highlight={st['collide']}")

    print(__doc__)
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.cam.lookat[:] = [0, 0, 1.2]
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 60, -28, 13
        last = time.time()
        last_label = None
        while v.is_running():
            now = time.time()
            if not st["paused"]:
                st["t"] = min(st["t"] + (now - last) * st["speed"], pb.duration)
            last = now
            with v.lock():
                pb.apply(st["t"])
                if st["collide"]:
                    highlight_collision(m, pb.cc, base)
                v.user_scn.ngeom = 0
                pb.decorate(v.user_scn, st["t"])
            lab = pb.label(st["t"])
            if lab != last_label:
                print(f"[{st['t']:6.1f} s] {lab}")
                last_label = lab
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
