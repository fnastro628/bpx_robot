"""
Heel behavior — follow the owner at a fixed distance.

The robot maintains TARGET_SIZE as the person's apparent bbox height.
A simple proportional controller adjusts forward speed; yaw keeps the
person centred in frame.  A deadband prevents oscillation when at distance.

Runs indefinitely until cancelled by a new command.
"""

import asyncio
import json

YAW_GAIN      = 1.2    # rad/s per unit cx_norm error
FOLLOW_SPEED  = 0.4    # max forward speed (m/s)
TARGET_SIZE   = 0.35   # desired bbox height fraction (≈ 0.8 m behind)
DEADBAND      = 0.05   # size error deadband
LOOP_HZ       = 20     # control loop rate


class HeelBehavior:
    def __init__(self, manager):
        self._mgr = manager
        self._det = None

        from std_msgs.msg import String
        self._sub = manager.create_subscription(
            String, "/perception/person_detections",
            lambda msg: setattr(self, "_det", json.loads(msg.data)), 10
        )

    async def run(self, **_kwargs):
        self._mgr.get_logger().info("HEEL: following owner")

        try:
            while True:
                det = self._det

                if det is None:
                    self._mgr.send_velocity(0.0, 0.0, 0.0)
                    await asyncio.sleep(1.0 / LOOP_HZ)
                    continue

                cx      = det["cx"]
                size    = det["size"]
                err     = TARGET_SIZE - size   # positive → too far, negative → too close

                yaw = -YAW_GAIN * cx

                if abs(err) < DEADBAND:
                    fwd = 0.0
                else:
                    # Proportional: scale speed by normalised distance error
                    fwd = FOLLOW_SPEED * (err / TARGET_SIZE)
                    fwd = max(-0.25, min(FOLLOW_SPEED, fwd))

                self._mgr.send_velocity(fwd, 0.0, yaw)
                await asyncio.sleep(1.0 / LOOP_HZ)

        except asyncio.CancelledError:
            self._mgr.send_velocity(0.0, 0.0, 0.0)
            raise
