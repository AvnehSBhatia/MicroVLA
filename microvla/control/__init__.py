"""Structured control: learned world goals + the teacher's proven servo shell.

See ``paper/WHY_THE_TEACHER_WORKS.md`` for the design rationale and
``goal_head.py`` / ``machine.py`` for the two halves: what is LEARNED
(task content: where to grasp, where to place) and what is STRUCTURE
(control: latch, one-way phases, P-law, hold check, probe search).
"""
from microvla.control.goal_head import (FEATURE_VERSION, GraspPointHead,
                                        PlaceHead, build_grasp_features,
                                        load_goal_heads, save_goal_heads,
                                        yaw_from_quat)
from microvla.control.machine import GoalServoMachine, PROBE_XY, PROBE_YAW

__all__ = [
    "FEATURE_VERSION", "GraspPointHead", "PlaceHead", "build_grasp_features",
    "load_goal_heads", "save_goal_heads", "yaw_from_quat",
    "GoalServoMachine", "PROBE_XY", "PROBE_YAW",
]
