import agplib
import pytest
import numpy as np

def test_version():
    assert hasattr(agplib, "__version__")

def test_attributes():
    assert hasattr(agplib, "simple_test")

def simple_test():
    assert agplib.simple_test(2) == 4
    