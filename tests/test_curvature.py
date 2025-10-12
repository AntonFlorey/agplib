import agplib
import pytest
import numpy as np

def test_planar_quad_curvature():
    # simple test with a planar quad
    vertices_with_normals = [
        agplib.VertexWithNormal([0,0,0], [0,0,1]),
        agplib.VertexWithNormal([1,0,0], [0,0,1]),
        agplib.VertexWithNormal([1,1,0], [0,0,1]),
        agplib.VertexWithNormal([0,1,0], [0,0,1])
    ]

    tangent_space = agplib.TangentSpace([1,0,0], [0,1,0])

    IIff = agplib.approximate_II_fundamental_form(vertices_with_normals, tangent_space)

    print("II fundamental form:", IIff)

    principal_curvature_info_v1 = agplib.compute_principal_curvature(vertices_with_normals, tangent_space)
    principal_curvature_info_v2 = agplib.compute_principal_curvature(vertices_with_normals, np.asarray([0,0,1]))

    print("Principal curvature direction info:", principal_curvature_info_v1.direction, principal_curvature_info_v1.unambiguity)
    print("The same info:", principal_curvature_info_v2.direction, principal_curvature_info_v2.unambiguity)

    assert np.allclose(principal_curvature_info_v1.unambiguity, principal_curvature_info_v2.unambiguity)

def test_curved_quad_curvature():
    # simple test with a quad with some positive curvature in x-direction
    rot_angle = np.deg2rad(20)
    vertices_with_normals = [
        agplib.VertexWithNormal([0,0,0], [-np.sin(rot_angle),0,np.cos(rot_angle)]),
        agplib.VertexWithNormal([1,0,0], [np.sin(rot_angle),0,np.cos(rot_angle)]),
        agplib.VertexWithNormal([1,1,0], [np.sin(rot_angle),0,np.cos(rot_angle)]),
        agplib.VertexWithNormal([0,1,0], [-np.sin(rot_angle),0,np.cos(rot_angle)])
    ]

    tangent_space = agplib.TangentSpace([1,0,0], [0,1,0])

    IIff = agplib.approximate_II_fundamental_form(vertices_with_normals, tangent_space)

    print("II fundamental form:", IIff)

    principal_curvature_info_v1 = agplib.compute_principal_curvature(vertices_with_normals, tangent_space)
    principal_curvature_info_v2 = agplib.compute_principal_curvature(vertices_with_normals, np.asarray([0,0,1]))

    print("Principal curvature direction info:", principal_curvature_info_v1.direction, principal_curvature_info_v1.unambiguity)
    print("The same info:", principal_curvature_info_v2.direction, principal_curvature_info_v2.unambiguity)

    assert np.allclose(principal_curvature_info_v1.unambiguity, principal_curvature_info_v2.unambiguity)
    assert np.allclose(1.0, np.abs(np.dot(principal_curvature_info_v1.direction, [0,1,0])))
    assert np.allclose(1.0, np.abs(np.dot(principal_curvature_info_v2.direction, [0,1,0])))