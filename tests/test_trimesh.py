import agplib
import pytest
import numpy as np

def test_splitted_quad():
    quad = agplib.TriMesh(vertex_positions=[[0,0,0], [1,0,0], [1,1,0], [0,1,0]],
                          vertex_normals=[[0,0,1], [0,0,1], [0,0,1], [0,0,1]],
                          edge_vertices=[0,1,1,2,2,3,3,0,0,2],
                          face_normals=[[0,0,1], [0,0,1]],
                          face_vertices=[0,1,2,2,3,0])
    
    whole_quad = agplib.grow_face_subset_to_disc([0], quad, 1)
    assert whole_quad.face_set.faces == {0,1}
    assert whole_quad.face_set.vertices == {0,1,2,3}
    assert whole_quad.face_set.edges == {0,1,2,3,4}
    assert set(whole_quad.boundary_vertices) == {0,1,2,3}
