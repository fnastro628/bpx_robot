"""
Mirrors the SDK enums from motion_types.h and maps natural-language phrases
to gaits and command actions.  Import this module from any Python node.
"""

from enum import IntEnum
from typing import Dict, Tuple, Union


class MotionState(IntEnum):
    LYING_DOWN   = 0
    STANDING_UP  = 1
    PASSIVE      = 2
    SIT_DOWN     = 3
    MOTION       = 6   # robot is up and can receive velocity commands


class MotionGait(IntEnum):
    WALK          = 0
    BIPEDAL       = 3
    FLIP          = 4
    WALK_PHASE    = 6
    POSE_TRACKING = 7
    RUNNING       = 8
    WALK_PERIOD   = 10


# Maps speech phrase → MotionGait
GAIT_ALIASES: Dict[str, MotionGait] = {
    "walk":          MotionGait.WALK,
    "walking":       MotionGait.WALK,
    "trot":          MotionGait.WALK,
    "stroll":        MotionGait.WALK,
    "run":           MotionGait.RUNNING,
    "running":       MotionGait.RUNNING,
    "sprint":        MotionGait.RUNNING,
    "bipedal":       MotionGait.BIPEDAL,
    "two legs":      MotionGait.BIPEDAL,
    "stand tall":    MotionGait.BIPEDAL,
    "flip":          MotionGait.FLIP,
    "backflip":      MotionGait.FLIP,
    "back flip":     MotionGait.FLIP,
    "walk phase":    MotionGait.WALK_PHASE,
    "phase walk":    MotionGait.WALK_PHASE,
    "pose":          MotionGait.POSE_TRACKING,
    "pose tracking": MotionGait.POSE_TRACKING,
    "periodic":      MotionGait.WALK_PERIOD,
    "walk period":   MotionGait.WALK_PERIOD,
}

# Command action type:  str  = simple action name
#                       tuple = ("velocity", {x, y, yaw})
CommandAction = Union[str, Tuple[str, Dict[str, float]]]

COMMAND_ALIASES: Dict[str, CommandAction] = {
    # ── State transitions ────────────────────────────────────────────────────
    "stand":          "stand",
    "stand up":       "stand",
    "get up":         "stand",
    "on your feet":   "stand",
    "sit":            "sit",
    "sit down":       "sit",
    "lie down":       "lie_down",
    "lay down":       "lie_down",
    "stop":           "stop",
    "halt":           "stop",
    "freeze":         "stop",
    "stay":           "stop",
    "wait":           "stop",
    "damp":           "damp",
    "relax":          "damp",
    "power off":      "damp",
    "sleep":          "damp",

    # ── Behavior primitives ───────────────────────────────────────────────────
    "come":           "come",
    "come here":      "come",
    "here":           "come",
    "come to me":     "come",
    "heel":           "heel",
    "follow me":      "heel",
    "follow":         "heel",
    "walk with me":   "heel",
    "fetch":          "fetch",
    "get it":         "fetch",
    "retrieve":       "fetch",
    "bring it":       "fetch",
    "find":           "find",
    "search":         "find",
    "look for":       "find",
    "patrol":         "patrol",

    # ── Directional velocity (x m/s, y m/s, yaw rad/s) ───────────────────────
    "forward":        ("velocity", {"x":  0.5, "y":  0.0, "yaw":  0.0}),
    "go forward":     ("velocity", {"x":  0.5, "y":  0.0, "yaw":  0.0}),
    "advance":        ("velocity", {"x":  0.5, "y":  0.0, "yaw":  0.0}),
    "back":           ("velocity", {"x": -0.5, "y":  0.0, "yaw":  0.0}),
    "backward":       ("velocity", {"x": -0.5, "y":  0.0, "yaw":  0.0}),
    "go back":        ("velocity", {"x": -0.5, "y":  0.0, "yaw":  0.0}),
    "reverse":        ("velocity", {"x": -0.5, "y":  0.0, "yaw":  0.0}),
    "left":           ("velocity", {"x":  0.0, "y":  0.3, "yaw":  0.0}),
    "strafe left":    ("velocity", {"x":  0.0, "y":  0.3, "yaw":  0.0}),
    "right":          ("velocity", {"x":  0.0, "y": -0.3, "yaw":  0.0}),
    "strafe right":   ("velocity", {"x":  0.0, "y": -0.3, "yaw":  0.0}),
    "turn left":      ("velocity", {"x":  0.0, "y":  0.0, "yaw":  0.6}),
    "turn right":     ("velocity", {"x":  0.0, "y":  0.0, "yaw": -0.6}),
    "spin":           ("velocity", {"x":  0.0, "y":  0.0, "yaw":  1.2}),
    "spin left":      ("velocity", {"x":  0.0, "y":  0.0, "yaw":  1.2}),
    "spin right":     ("velocity", {"x":  0.0, "y":  0.0, "yaw": -1.2}),
    "rotate":         ("velocity", {"x":  0.0, "y":  0.0, "yaw":  0.8}),
}
