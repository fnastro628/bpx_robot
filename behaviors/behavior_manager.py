#!/usr/bin/env python3
"""
Central behavior orchestrator.

Subscribes to /commands (JSON from command_parser).
Routes commands to direct robot actions (via BPX services / /cmd_vel)
or to long-running behavior coroutines (come, heel, fetch, patrol).

Running behaviors can be interrupted at any time by a new command.
Uses asyncio inside a ROS2 MultiThreadedExecutor for non-blocking behavior.
"""

import asyncio
import json
import threading
import sys
import os

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from std_msgs.msg import String, UInt8
from std_srvs.srv import Trigger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from behaviors.gait_map import MotionState
from behaviors.primitives.come  import ComeBehavior
from behaviors.primitives.heel  import HeelBehavior
from behaviors.primitives.fetch import FetchBehavior


class BehaviorManager(Node):
    def __init__(self):
        super().__init__("behavior_manager")

        cbg = ReentrantCallbackGroup()

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            UInt8, "/bpx/motion_state", self._on_state, 10, callback_group=cbg
        )
        self.create_subscription(
            String, "/commands", self._on_command, 10, callback_group=cbg
        )

        # ── BPX service clients ────────────────────────────────────────────────
        self.stand_cli = self.create_client(Trigger, "/bpx/stand")
        self.sit_cli   = self.create_client(Trigger, "/bpx/sit")
        self.damp_cli  = self.create_client(Trigger, "/bpx/damp")

        # ── State ─────────────────────────────────────────────────────────────
        self.motion_state: int = MotionState.LYING_DOWN
        self._behavior_task: asyncio.Task | None = None

        # ── Behavior instances ────────────────────────────────────────────────
        self._behaviors = {
            "come":  ComeBehavior(self),
            "heel":  HeelBehavior(self),
            "fetch": FetchBehavior(self),
        }

        # asyncio event loop running in a background thread
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

        self.get_logger().info("Behavior manager ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_state(self, msg: UInt8):
        self.motion_state = msg.data

    def _on_command(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Bad command JSON: {msg.data}")
            return
        self.get_logger().info(f"Dispatching: {cmd}")
        asyncio.run_coroutine_threadsafe(self._dispatch(cmd), self._loop)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, cmd: dict):
        await self._cancel_behavior()

        ctype = cmd.get("type")

        if ctype == "stand":
            await self._call(self.stand_cli)

        elif ctype == "sit":
            await self._call(self.sit_cli)

        elif ctype in ("damp", "relax"):
            await self._call(self.damp_cli)

        elif ctype == "stop":
            self.send_velocity(0.0, 0.0, 0.0)

        elif ctype == "velocity":
            self.send_velocity(
                cmd.get("x", 0.0), cmd.get("y", 0.0), cmd.get("yaw", 0.0)
            )

        elif ctype == "gait":
            # TODO: implement a /bpx/set_gait service in the C++ driver
            self.get_logger().info(f"Gait change → {cmd.get('gait')} (not yet wired)")

        elif ctype in self._behaviors:
            behavior = self._behaviors[ctype]
            kwargs = {k: v for k, v in cmd.items() if k != "type"}
            self._behavior_task = asyncio.ensure_future(behavior.run(**kwargs))

        elif ctype == "patrol":
            self._behavior_task = asyncio.ensure_future(self._patrol())

        elif ctype == "event":
            await self._handle_acoustic_event(cmd)

        elif ctype == "page_approach":
            self._behavior_task = asyncio.ensure_future(
                self._approach_by_doa(cmd.get("doa_deg", 0.0))
            )

        else:
            self.get_logger().warn(f"Unknown command type: '{ctype}'")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _cancel_behavior(self):
        if self._behavior_task and not self._behavior_task.done():
            self._behavior_task.cancel()
            try:
                await self._behavior_task
            except asyncio.CancelledError:
                pass
            self.send_velocity(0.0, 0.0, 0.0)

    async def _call(self, client):
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"Service {client.srv_name} not available")
            return
        future = client.call_async(Trigger.Request())
        await asyncio.wrap_future(future)

    def send_velocity(self, x: float, y: float, yaw: float):
        twist = Twist()
        twist.linear.x  = float(x)
        twist.linear.y  = float(y)
        twist.angular.z = float(yaw)
        self.cmd_vel_pub.publish(twist)

    # ── Acoustic event handler ────────────────────────────────────────────────

    async def _handle_acoustic_event(self, cmd: dict):
        event = cmd.get("event", "")
        doa   = cmd.get("doa_deg", 0.0)
        self.get_logger().info(f"Acoustic event: {event} from {doa:.0f}°")

        if event == "glass_break":
            self.get_logger().warn("ALERT: glass break detected!")
            # Navigate toward the sound source
            self._behavior_task = asyncio.ensure_future(self._approach_by_doa(doa))

        elif event == "alarm":
            self.get_logger().error("ALERT: smoke/fire alarm detected!")
            self.send_velocity(0.0, 0.0, 0.0)
            # TODO: trigger TTS alert and notify owner via network

        elif event == "baby_cry":
            self.get_logger().warn("Baby cry detected — navigating to source.")
            self._behavior_task = asyncio.ensure_future(self._approach_by_doa(doa))

        elif event == "fall":
            self.get_logger().warn("Possible fall detected — checking on person.")
            self._behavior_task = asyncio.ensure_future(self._approach_by_doa(doa))

        elif event == "loud_impact":
            self.get_logger().warn("Loud impact detected.")

    async def _approach_by_doa(self, doa_deg: float):
        """Turn toward a DOA angle then use visual/acoustic come behavior."""
        from geometry_msgs.msg import Twist
        # First: turn to face the DOA direction
        yaw_rate  = 0.6 if doa_deg > 0 else -0.6
        turn_secs = abs(doa_deg) / 60.0   # rough estimate at 60°/s yaw
        self.send_velocity(0.0, 0.0, yaw_rate)
        await asyncio.sleep(min(turn_secs, 3.0))
        self.send_velocity(0.0, 0.0, 0.0)
        # Then hand off to come behavior which uses vision
        await self._behaviors["come"].run()

    # ── Built-in patrol ───────────────────────────────────────────────────────

    async def _patrol(self):
        """Simple square patrol: forward → turn × 4."""
        self.get_logger().info("PATROL: starting square loop")
        speed, turn_speed = 0.4, 0.6
        try:
            while True:
                for _ in range(4):
                    self.send_velocity(speed, 0.0, 0.0)
                    await asyncio.sleep(3.0)
                    self.send_velocity(0.0, 0.0, turn_speed)
                    await asyncio.sleep(2.35)   # ≈ 90° turn
        except asyncio.CancelledError:
            self.send_velocity(0.0, 0.0, 0.0)
            raise


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = BehaviorManager()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
