"""Gravity and Quaternion Tests"""
import pytest
import numpy as np

from plav.simulator import get_gravity
from plav.quaternion_math import to_euler

def test_get_gravity():
    """Test the get_gravity function"""
    phi = 0.0  # radians
    h = 0.0   # meters
    gravity = get_gravity(phi, h) #get_gravity(phi, h)

    # gravity hella wrong
    assert  9.0 < gravity < 10.0, "Gravity totally wrong or wrong units"
    # gravity is set to constant, wrong
    assert  9.78 < gravity < 9.79, "Gravity calculation not J_2"

    gravity = get_gravity(1.57, h) #get_gravity(phi, h)
    assert 9.83 < gravity < 9.84, "Gravity doesn't change with latitude"

    gravity = get_gravity(1.57, 30000.0)
    assert 9.73 < gravity < 9.74, "Gravity doesn't change with altitude"

def test_quaternion_standard():
    """Test that quaternions use first element as scalar"""

    # identity quaternion
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    euler = to_euler(q_identity)
    assert np.allclose(euler, np.array([0.0, 0.0, 0.0])), \
        "Identity quaternion did not convert to zero euler angles, check quaternion convention"

    # 90 degree rotation about z axis
    q_90_z = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
    euler = to_euler(q_90_z)
    assert np.allclose(euler, np.array([0.0, 0.0, np.pi/2])), \
        "90 degree rotation about z axis did not convert to expected euler angles"
