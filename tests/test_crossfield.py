import agplib.crossfield as cf
import pytest
import numpy as np

def test_simple_cube():
    surface_graph = cf.SurfaceGraph()
    node_id_0 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, np.array([0,0,1])))
    node_id_1 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, np.array([1,0,0])))
    node_id_2 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, np.array([0,-1,0])))
    node_id_3 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, np.array([-1,0,0])))
    node_id_4 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, np.array([0,1,0])))
    node_id_5 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, np.array([0,0,-1])))
    surface_graph.add_edge(node_id_0, node_id_1)
    surface_graph.add_edge(node_id_0, node_id_2)
    surface_graph.add_edge(node_id_0, node_id_3)
    surface_graph.add_edge(node_id_0, node_id_4)
    surface_graph.add_edge(node_id_5, node_id_1)
    surface_graph.add_edge(node_id_5, node_id_2)
    surface_graph.add_edge(node_id_5, node_id_3)
    surface_graph.add_edge(node_id_5, node_id_4)
    surface_graph.add_edge(node_id_1, node_id_2)
    surface_graph.add_edge(node_id_2, node_id_3)
    surface_graph.add_edge(node_id_3, node_id_4)
    surface_graph.add_edge(node_id_4, node_id_1)

    crossfield = cf.compute_crossfield(surface_graph, [])
    print("Crossfield of a unit cube:", crossfield)

def test_slanted_cube():
    # this object can not exist in real life but who cares...
    surface_graph = cf.SurfaceGraph()
    magic_val = 1 / np.sqrt(2)
    normals = [np.array([0,0,1]), 
               np.array([magic_val,0,magic_val]),
               np.array([0,-magic_val,magic_val]),
               np.array([-magic_val,0,magic_val]),
               np.array([0,magic_val,magic_val]),
               np.array([0,0,-1])]
    
    node_id_0 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, normals[0]))
    node_id_1 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, normals[1]))
    node_id_2 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, normals[2]))
    node_id_3 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, normals[3]))
    node_id_4 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, normals[4]))
    node_id_5 = surface_graph.add_node(cf.SurfaceGraphNode(1.0, normals[5]))
    surface_graph.add_edge(node_id_0, node_id_1)
    surface_graph.add_edge(node_id_0, node_id_2)
    surface_graph.add_edge(node_id_0, node_id_3)
    surface_graph.add_edge(node_id_0, node_id_4)
    surface_graph.add_edge(node_id_5, node_id_1)
    surface_graph.add_edge(node_id_5, node_id_2)
    surface_graph.add_edge(node_id_5, node_id_3)
    surface_graph.add_edge(node_id_5, node_id_4)
    surface_graph.add_edge(node_id_1, node_id_2)
    surface_graph.add_edge(node_id_2, node_id_3)
    surface_graph.add_edge(node_id_3, node_id_4)
    surface_graph.add_edge(node_id_4, node_id_1)

    test_weight = 1e+5
    test_constraint_0 = cf.CrossConstraint(test_weight, np.array([1,0,0]), 0)
    test_constraint_1 = cf.CrossConstraint(test_weight, np.array([0,1,0]), 1)

    crossfield = cf.compute_crossfield(surface_graph, [test_constraint_0, test_constraint_1], max_iters=100, merge_normal_dot_th=0.2, max_multires_layers=0)
    print("Crossfield of a slanted cube:")
    for i in range(6):
        print(crossfield[i])
        assert np.allclose(0, np.dot(crossfield[i], normals[i]))
        assert np.allclose(1, np.linalg.norm(crossfield[i]))
    assert np.allclose(1.0, np.abs(np.dot(np.array([1,0,0]), crossfield[0]))) or np.allclose(1.0, np.abs(np.dot(np.array([0,1,0]), crossfield[0])))
    