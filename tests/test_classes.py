import agplib
import pytest
import numpy as np

def test_tangent_space():
    test_space_1 = agplib.TangentSpace()
    test_space_2 = agplib.TangentSpace(np.array([0,1,0]), np.array([1,0,0]))
    test_space_3 = agplib.TangentSpace(np.array([0,1,0]), np.array([1,0,0]), np.array([1,1,1]))
    print("Created tangent spaces:", type(test_space_1), type(test_space_2), type(test_space_3))

def test_vertex_with_normal():
    test_vertex = agplib.VertexWithNormal(np.array([1,2,0]), np.array([1,1,1]))
    print("Created vertex with normal:", type(test_vertex))

def test_surface_graph_node():
    test_area = 1.0
    test_normal = np.array([0,0,1])
    test_node = agplib.SurfaceGraphNode(test_area, test_normal)
    print("Created surface graph node:", type(test_node))

def test_surface_graph():
    surface_graph = agplib.SurfaceGraph()
    assert surface_graph.number_of_nodes() == 0
    node_id_0 = surface_graph.add_node(agplib.SurfaceGraphNode(1.0, np.array([0,0,1])))
    node_id_1 = surface_graph.add_node(agplib.SurfaceGraphNode(1.0, np.array([0,0,1])))
    assert node_id_0 == 0 and node_id_1 == 1
    assert surface_graph.number_of_nodes() == 2
    surface_graph.add_edge(node_id_0, node_id_1)
    
def test_cross_constraint():
    test_direction = np.array([1,0,0])
    test_weight = 1.0
    test_node_id = 0
    test_constraint = agplib.CrossConstraint(test_weight, test_direction, test_node_id)
    print("Created a cross constraint:", type(test_constraint))