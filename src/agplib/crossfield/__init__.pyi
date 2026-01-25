from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing

__all__ = [
    'CrossConstaint', 
    'PrincipalCurvatureInfo', 
    'SurfaceGraph', 
    'SurfaceGraphNode', 
    'VertexWithNormal', 
    'approximate_II_fundamental_form', 
    'compute_crossfield', 
    'compute_principal_curvature'
    ]

class CrossConstraint:
    def __init__(self, weight: typing.SupportsFloat, direction: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], surface_node_id: typing.SupportsInt) -> None:
        ...
class PrincipalCurvatureInfo:
    @property
    def direction(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        ...
    @property
    def unambiguity(self) -> float:
        ...
class SurfaceGraph:
    def __init__(self) -> None:
        ...
    def add_edge(self, node_id_A: typing.SupportsInt, node_id_B: typing.SupportsInt) -> None:
        ...
    def add_node(self, node: SurfaceGraphNode) -> int:
        ...
    def number_of_nodes(self) -> int:
        ...
class SurfaceGraphNode:
    def __init__(self, area: typing.SupportsFloat, normal: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
class VertexWithNormal:
    def __init__(self, pos: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], normal: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
def approximate_II_fundamental_form(arg0: collections.abc.Sequence[VertexWithNormal], arg1: agplib._agplib.TangentSpace) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 2]"]:
    ...
def compute_crossfield(surface: SurfaceGraph, constraints: collections.abc.Sequence[CrossConstraint], max_iters: typing.SupportsInt = 20, max_multires_layers: typing.SupportsInt = 100, merge_normal_dot_th: typing.SupportsFloat = 0.5, convergence_eps: typing.SupportsFloat = 1e-06) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]]:
    ...
@typing.overload
def compute_principal_curvature(face_verts: collections.abc.Sequence[VertexWithNormal], tangent_space: agplib._agplib.TangentSpace) -> PrincipalCurvatureInfo:
    ...
@typing.overload
def compute_principal_curvature(face_verts: collections.abc.Sequence[VertexWithNormal], face_normal: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> PrincipalCurvatureInfo:
    ...
