import agplib
import pytest
import numpy as np
import agplib.crossfield as cf
import agplib.util as agputil

def test_version():
    assert hasattr(agplib, "__version__")

def test_convex_hull():
    test_points_A = np.array([[0,0], [1,0], [1,1], [0,1], [2, 0.4], [2, 0.4]])
    test_points_B = np.array([[0,0], [2,0], [2,1], [0,1], [1,0], [1.5,0], [2,0.4], [2,0.6], [2 - 1e-5, 0.2]])
    agputil.compute_convex_hull(test_points_A)
    agputil.compute_convex_hull(test_points_B)
    