import agplib
import pytest
import numpy as np
import agplib.crossfield as cf
import agplib.util as agputil

def test_version():
    assert hasattr(agplib, "__version__")

def test_convex_hull():
    for _ in range(1000):
        random_hull = agputil.compute_convex_hull(list(np.random.rand(100,2)))
