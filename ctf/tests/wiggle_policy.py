"""Deterministic test policy: fast, turning, jumping and sending sonar."""

import math


def decide(observation):
    tick, me = observation["tick"], observation["self"]
    phase = tick // 25 + me["id"]
    throttle = 1.0 if phase % 3 else -0.4
    sonar = (me["id"], tick) if phase % 4 == 0 else None
    return throttle, math.sin(0.07 * tick + me["id"]), phase % 7 == 0, sonar
