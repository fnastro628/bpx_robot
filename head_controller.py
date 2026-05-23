#!/usr/bin/env python3
"""
Shared head pan/tilt controller — Dynamixel XL430-W250.

Same ROS2 interface as before — all acoustic and vision nodes are unchanged.

Subscribes:
  /head/pan_deg         Float32  — target pan  (°), clamped to config range
  /head/tilt_deg        Float32  — target tilt (°), clamped to config range

Publishes:
  /head/pan_actual_deg  Float32  — current encoder-read pan  (°)
  /head/tilt_actual_deg Float32  — current encoder-read tilt (°)

Speed limiting is handled by the XL430's built-in Profile Velocity register,
set once at startup. Goal positions are written at 50 Hz; the servo executes
a trapezoidal motion profile internally.

Parameters:
  port               (string)    U2D2 serial port — default /dev/ttyUSB0
  baudrate           (int)       Dynamixel baud   — default 57600
  pan_id             (int)       Dynamixel servo ID for pan  — default 1
  tilt_id            (int)       Dynamixel servo ID for tilt — default 2
  pan_range_deg      ([lo, hi])  Clamp limits in degrees     — default [-90, 90]
  tilt_range_deg     ([lo, hi])  Clamp limits in degrees     — default [-30, 40]
  pan_center_deg     (float)     Physical angle that equals 0° pan  — default 180.0
  tilt_center_deg    (float)     Physical angle that equals 0° tilt — default 180.0
  max_pan_speed_dps  (float)     Max pan  speed °/s — default 120.0
  max_tilt_speed_dps (float)     Max tilt speed °/s — default 60.0

Wiring:
  U2D2 → JST 3-pin TTL chain → XL430 pan (ID 1) → XL430 tilt (ID 2)
  Power: 3S LiPo from body, routed through neck cable

Requires:
  pip install dynamixel-sdk
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# ── XL430-W250 Control Table (Protocol 2.0) ───────────────────────────────────
ADDR_OPERATING_MODE    = 11
ADDR_MIN_POS_LIMIT     = 52
ADDR_MAX_POS_LIMIT     = 48
ADDR_TORQUE_ENABLE     = 64
ADDR_PROFILE_VELOCITY  = 112
ADDR_GOAL_POSITION     = 116
ADDR_PRESENT_POSITION  = 132

PROTOCOL_VERSION  = 2.0
POSITION_MODE     = 3       # operating mode: position control
TORQUE_ENABLE     = 1
TORQUE_DISABLE    = 0

# XL430 resolution: 4096 steps over 360°
STEPS_PER_DEG = 4096.0 / 360.0
DEG_PER_STEP  = 360.0 / 4096.0

# Profile velocity unit: 0.229 RPM per unit
RPM_PER_UNIT  = 0.229


def _dps_to_profile_velocity(dps: float) -> int:
    """Convert degrees/second to XL430 Profile Velocity register value."""
    rpm   = dps / 360.0 * 60.0
    units = int(rpm / RPM_PER_UNIT)
    return max(1, min(units, 32767))   # 0 = unlimited; clamp to valid range


def _deg_to_position(angle_deg: float, center_deg: float) -> int:
    """Map a logical angle (° from centre) to a raw XL430 position count."""
    raw = center_deg + angle_deg
    return int(round(raw * STEPS_PER_DEG)) & 0xFFFF


def _position_to_deg(position: int, center_deg: float) -> float:
    """Map a raw XL430 position count to logical angle (° from centre)."""
    return position * DEG_PER_STEP - center_deg


class HeadController(Node):
    def __init__(self):
        super().__init__("head_controller")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("port",               "/dev/ttyUSB0")
        self.declare_parameter("baudrate",           57600)
        self.declare_parameter("pan_id",             1)
        self.declare_parameter("tilt_id",            2)
        self.declare_parameter("pan_range_deg",      [-90.0,  90.0])
        self.declare_parameter("tilt_range_deg",     [-30.0,  40.0])
        self.declare_parameter("pan_center_deg",     180.0)
        self.declare_parameter("tilt_center_deg",    180.0)
        self.declare_parameter("max_pan_speed_dps",  120.0)
        self.declare_parameter("max_tilt_speed_dps",  60.0)

        port           = self.get_parameter("port").value
        baudrate       = self.get_parameter("baudrate").value
        self._pan_id   = self.get_parameter("pan_id").value
        self._tilt_id  = self.get_parameter("tilt_id").value
        pan_range      = self.get_parameter("pan_range_deg").value
        tilt_range     = self.get_parameter("tilt_range_deg").value
        self._pan_min  = float(pan_range[0])
        self._pan_max  = float(pan_range[1])
        self._tilt_min = float(tilt_range[0])
        self._tilt_max = float(tilt_range[1])
        self._pan_ctr  = self.get_parameter("pan_center_deg").value
        self._tilt_ctr = self.get_parameter("tilt_center_deg").value
        pan_vel        = _dps_to_profile_velocity(
                             self.get_parameter("max_pan_speed_dps").value)
        tilt_vel       = _dps_to_profile_velocity(
                             self.get_parameter("max_tilt_speed_dps").value)

        # ── Targets (in logical °, clamped) ───────────────────────────────────
        self._tgt_pan  = 0.0
        self._tgt_tilt = 0.0

        # ── Hardware ──────────────────────────────────────────────────────────
        self._port_h, self._pkt_h = self._init_dynamixel(port, baudrate)
        self._sim = (self._port_h is None)

        if not self._sim:
            for sid, vel, lo, hi in [
                (self._pan_id,   pan_vel,  self._pan_min,  self._pan_max),
                (self._tilt_id,  tilt_vel, self._tilt_min, self._tilt_max),
            ]:
                self._configure_servo(sid, vel, lo, hi)
            # Move to centre on startup
            self._write_goal(self._pan_id,  self._tgt_pan,  self._pan_ctr)
            self._write_goal(self._tilt_id, self._tgt_tilt, self._tilt_ctr)

        # ── ROS2 interface ────────────────────────────────────────────────────
        self._pan_actual_pub  = self.create_publisher(
            Float32, "/head/pan_actual_deg",  10)
        self._tilt_actual_pub = self.create_publisher(
            Float32, "/head/tilt_actual_deg", 10)

        self.create_subscription(Float32, "/head/pan_deg",  self._on_pan,  10)
        self.create_subscription(Float32, "/head/tilt_deg", self._on_tilt, 10)

        self.create_timer(0.02, self._update)   # 50 Hz

        self.get_logger().info(
            f"HeadController ready ({'SIMULATION' if self._sim else port}) — "
            f"pan ID {self._pan_id} [{self._pan_min}°,{self._pan_max}°], "
            f"tilt ID {self._tilt_id} [{self._tilt_min}°,{self._tilt_max}°]"
        )

    # ── Hardware init ─────────────────────────────────────────────────────────

    def _init_dynamixel(self, port: str, baudrate: int):
        try:
            from dynamixel_sdk import PortHandler, PacketHandler
            ph = PortHandler(port)
            if not ph.openPort():
                raise OSError(f"Cannot open {port}")
            if not ph.setBaudRate(baudrate):
                raise OSError(f"Cannot set baudrate {baudrate}")
            pkt = PacketHandler(PROTOCOL_VERSION)
            self.get_logger().info(
                f"Dynamixel port open: {port} @ {baudrate} baud"
            )
            return ph, pkt
        except Exception as e:
            self.get_logger().warn(
                f"Dynamixel not available ({e}). Running in simulation mode."
            )
            return None, None

    def _configure_servo(self, sid: int, profile_vel: int,
                          lo_deg: float, hi_deg: float):
        """Set operating mode, position limits, profile velocity, enable torque."""
        write1 = self._pkt_h.write1ByteTxRx
        write4 = self._pkt_h.write4ByteTxRx
        ph     = self._port_h

        # Disable torque first — required before changing operating mode / limits
        write1(ph, sid, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

        # Position control mode
        write1(ph, sid, ADDR_OPERATING_MODE, POSITION_MODE)

        # Hardware position limits (in steps)
        lo_pos = _deg_to_position(lo_deg, self._pan_ctr if sid == self._pan_id
                                  else self._tilt_ctr)
        hi_pos = _deg_to_position(hi_deg, self._pan_ctr if sid == self._pan_id
                                  else self._tilt_ctr)
        write4(ph, sid, ADDR_MIN_POS_LIMIT, min(lo_pos, hi_pos))
        write4(ph, sid, ADDR_MAX_POS_LIMIT, max(lo_pos, hi_pos))

        # Profile velocity — Dynamixel enforces this on every move
        write4(ph, sid, ADDR_PROFILE_VELOCITY, profile_vel)

        # Enable torque
        write1(ph, sid, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        self.get_logger().info(
            f"  Servo ID {sid}: profile_vel={profile_vel}, "
            f"pos limits [{lo_pos}, {hi_pos}]"
        )

    # ── Subscriptions ─────────────────────────────────────────────────────────

    def _on_pan(self, msg: Float32):
        self._tgt_pan = max(self._pan_min, min(self._pan_max, float(msg.data)))

    def _on_tilt(self, msg: Float32):
        self._tgt_tilt = max(self._tilt_min, min(self._tilt_max, float(msg.data)))

    # ── 50 Hz update ──────────────────────────────────────────────────────────

    def _update(self):
        if not self._sim:
            self._write_goal(self._pan_id,  self._tgt_pan,  self._pan_ctr)
            self._write_goal(self._tilt_id, self._tgt_tilt, self._tilt_ctr)
            self._publish_actual()

    def _write_goal(self, sid: int, angle_deg: float, center_deg: float):
        pos = _deg_to_position(angle_deg, center_deg)
        result, error = self._pkt_h.write4ByteTxRx(
            self._port_h, sid, ADDR_GOAL_POSITION, pos
        )
        if result != 0:   # COMM_SUCCESS = 0
            self.get_logger().warn(
                f"Servo {sid} write failed: result={result} error={error}"
            )

    def _publish_actual(self):
        for sid, center, pub in [
            (self._pan_id,  self._pan_ctr,  self._pan_actual_pub),
            (self._tilt_id, self._tilt_ctr, self._tilt_actual_pub),
        ]:
            pos, result, _ = self._pkt_h.read4ByteTxRx(
                self._port_h, sid, ADDR_PRESENT_POSITION
            )
            if result == 0:
                msg      = Float32()
                msg.data = float(_position_to_deg(pos, center))
                pub.publish(msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        if not self._sim:
            for sid in (self._pan_id, self._tilt_id):
                self._pkt_h.write1ByteTxRx(
                    self._port_h, sid, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
                )
            self._port_h.closePort()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeadController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
