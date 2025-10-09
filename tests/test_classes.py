import agplib
import pytest
import numpy as np

def test_tangent_space():
    test_space = agplib.TangentSpace()
    print(type(test_space))

def test_vertex_with_normal():
    test_vertex = agplib.VertexWithNormal(np.array([1,2,0]), np.array([1,1,1]))
    print(type(test_vertex))