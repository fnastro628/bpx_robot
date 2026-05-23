"""
Fetch behavior — find a target object, approach it, return to owner.

Phases:
  1. SEARCH  — spin slowly until the target object appears in detections
  2. APPROACH — drive toward the object using visual servoing
  3. RETURN  — use ComeBehavior to navigate back to the owner

/perception/object_detections publishes the largest non-person detection as:
  { "class": str, "conf": float, "cx": float, "size": float, "bbox": [...] }

target_class=None matches any object (useful for "fetch whatever you see").
"""

import asyncio
import json

from behaviors.primitives.come import ComeBehavior

APPROACH_SPEED  = 0.30
YAW_GAIN        = 1.2
TOUCH_SIZE      = 0.55   # stop this close to the object
SEARCH_YAW      = 0.25
SEARCH_TIMEOUT  = 20.0
APPROACH_TIMEOUT = 30.0


class FetchBehavior:
    def __init__(self, manager):
        self._mgr        = manager
        self._obj        = None
        self._come       = ComeBehavior(manager)

        from std_msgs.msg import String
        self._sub = manager.create_subscription(
            String, "/perception/object_detections",
            lambda msg: setattr(self, "_obj", json.loads(msg.data)), 10
        )

    async def run(self, target: str | None = None, **_kwargs):
        self._mgr.get_logger().info(f"FETCH: looking for '{target or 'any object'}'")

        try:
            # Phase 1: search
            found = await self._search(target)
            if not found:
                self._mgr.get_logger().warn("FETCH: object not found — aborting")
                return

            # Phase 2: approach
            self._mgr.get_logger().info("FETCH: approaching object")
            await self._approach(target)

            # Phase 3: return
            self._mgr.get_logger().info("FETCH: returning to owner")
            await self._come.run()

        except asyncio.CancelledError:
            self._mgr.send_velocity(0.0, 0.0, 0.0)
            raise

    # ── Phases ────────────────────────────────────────────────────────────────

    async def _search(self, target: str | None) -> bool:
        deadline = asyncio.get_event_loop().time() + SEARCH_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            if self._obj_matches(target):
                return True
            self._mgr.send_velocity(0.0, 0.0, SEARCH_YAW)
            await asyncio.sleep(0.1)
        self._mgr.send_velocity(0.0, 0.0, 0.0)
        return False

    async def _approach(self, target: str | None):
        deadline = asyncio.get_event_loop().time() + APPROACH_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            if not self._obj_matches(target):
                await asyncio.sleep(0.1)
                continue

            obj  = self._obj
            size = obj["size"]
            cx   = obj["cx"]

            if size >= TOUCH_SIZE:
                self._mgr.send_velocity(0.0, 0.0, 0.0)
                return

            yaw = -YAW_GAIN * cx
            self._mgr.send_velocity(APPROACH_SPEED, 0.0, yaw)
            await asyncio.sleep(0.05)

        self._mgr.send_velocity(0.0, 0.0, 0.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _obj_matches(self, target: str | None) -> bool:
        if self._obj is None:
            return False
        if target is None:
            return True
        return self._obj.get("class", "").lower() == target.lower()
