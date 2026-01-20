"""
Anton's Geometry Processing Library: A small geometry processing package for the lazytopo Blender addon written in C++.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['CrossConstraint', 'PrincipalCurvatureInfo', 'SurfaceGraph', 'SurfaceGraphNode', 'TangentSpace', 'VertexWithNormal', 'approximate_II_fundamental_form', 'compute_crossfield', 'compute_principal_curvature', 'simple_test']
class CrossConstraint:
    def __init__(self, arg0: typing.SupportsFloat, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], arg2: typing.SupportsInt) -> None:
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
    def add_edge(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    def add_node(self, arg0: SurfaceGraphNode) -> int:
        ...
    def number_of_nodes(self) -> int:
        ...
class SurfaceGraphNode:
    def __init__(self, arg0: typing.SupportsFloat, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
class TangentSpace:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
class VertexWithNormal:
    def __init__(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
def approximate_II_fundamental_form(arg0: collections.abc.Sequence[VertexWithNormal], arg1: TangentSpace) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 2]"]:
    """
    Approximates the second fundamental form of a face with a least squares approach.
    """
def compute_crossfield(surface: SurfaceGraph, constraints: collections.abc.Sequence[CrossConstraint], max_iters: typing.SupportsInt = 10, max_multires_layers: typing.SupportsInt = 10, merge_normal_dot_th: typing.SupportsFloat = 0.5, convergence_eps: typing.SupportsFloat = 1e-06) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]]:
    """
    computes a smooth crossfield in the given surface
    """
@typing.overload
def compute_principal_curvature(arg0: collections.abc.Sequence[VertexWithNormal], arg1: TangentSpace) -> PrincipalCurvatureInfo:
    """
    computes the principal curvature direction and unambiguity score of a given face.
    """
@typing.overload
def compute_principal_curvature(arg0: collections.abc.Sequence[VertexWithNormal], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> PrincipalCurvatureInfo:
    """
    computes the principal curvature direction and unambiguity score of a given face.
    """
def simple_test(n: typing.SupportsInt) -> int:
    """
    A simple test function that prints Hello World and multiplies an integer by two.
    """
__version__: str = 'dev'
