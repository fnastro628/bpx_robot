"""
Come behavior — navigate toward the nearest detected person.

Strategy: visual servoing using the bounding-box centroid and size from
/perception/person_detections (published by the object detector node).

  cx_norm  : horizontal offset from image centre, range [-1, +1]
  size_norm: bbox height as a fraction of frame height, range [0, 1]

The robot spins slowly to search if no person is visible, then drives
toward the person while centering them in frame.  Stops when the person
fills STOP_SIZE fraction of the frame (≈ arm's reach).
"""

import asyncio
import json

APPROACH_SPEED  = 0.35   # m/s forward
YAW_GAIN        = 1.2    # rad/s per unit cx_norm
STOP_SIZE       = 0.50   # fraction of frame height — "close enough"
SEARCH_YAW      = 0.25   # rad/s slow spin while searching
TIMEOUT_SEC     = 30.0   # give up after this long


class ComeBehavior:
    def __init__(self, manager):
        self._mgr = manager
        self._det = None

        from std_msgs.msg import String
        self._sub = manager.create_subscription(
            String, "/perception/person_detections",
            lambda msg: setattr(self, "_det", json.loads(msg.data)), 10
        )

    async def run(self, **_kwargs):
        self._mgr.get_logger().info("COME: searching for person")
        deadline = asyncio.get_event_loop().time() + TIMEOUT_SEC

        while asyncio.get_event_loop().time() < deadline:
            det = self._det

            if det is None:
                # Spin slowly to search
                self._mgr.send_velocity(0.0, 0.0, SEARCH_YAW)
                await asyncio.sleep(0.1)
                continue

            cx   = det["cx"]          # -1..1
            size = det["size"]        # 0..1

            if size >= STOP_SIZE:
                self._mgr.send_velocity(0.0, 0.0, 0.0)
                self._mgr.get_logger().info("COME: arrived")
                return

            yaw = -YAW_GAIN * cx
            self._mgr.send_velocity(APPROACH_SPEED, 0.0, yaw)
            await asyncio.sleep(0.05)

        self._mgr.send_velocity(0.0, 0.0, 0.0)
        self._mgr.get_logger().warn("COME: timeout — person not reached")
