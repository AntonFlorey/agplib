"""
A(nton's) Geometry Processing Library: A small geometry processing package for the lazytopo Blender addon written in C++.
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
import collections.abc
from . import crossfield
from . import util

__all__ = [
    'TangentSpace', 
    'compute_any_tangent_space_basis', 
    'crossfield', 
    'map_point_to_world_space', 
    'util',
    "TriMesh",
    "DiscOnMesh",
    "FaceSetData",
    "flood_fill_face_subset_to_disc",
    "grow_face_subset_to_disc"
    ]
__version__: str = ...


class TangentSpace:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, x_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], y_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
    @typing.overload
    def __init__(self, x_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], y_axis: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> None:
        ...
def compute_any_tangent_space_basis(normal: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], origin: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"] = ...) -> TangentSpace:
    ...
def map_point_to_world_space(point: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"], tangent_space: TangentSpace) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    ...

class TriMesh:
    def __init__(self, 
                 vertex_positions: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]], 
                 vertex_normals: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]],
                 edge_vertices: collections.abc.Sequence[typing.SupportsInt],
                 face_normals: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]], 
                 face_vertices: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
class DiscOnMesh:
    @property
    def boundary_vertices(self) -> list[int]:
        ...
    @property
    def face_set(self) -> FaceSetData:
        ...
class FaceSetData:
    @property
    def edges(self) -> set[int]:
        ...
    @property
    def faces(self) -> set[int]:
        ...
    @property
    def vertices(self) -> set[int]:
        ...
def flood_fill_face_subset_to_disc(face_subset: collections.abc.Sequence[typing.SupportsInt], 
                                   mesh: TriMesh, start_face: typing.SupportsInt) -> DiscOnMesh | None:
    ...
def grow_face_subset_to_disc(face_subset: collections.abc.Sequence[typing.SupportsInt], 
                             mesh: TriMesh, max_grow_cycles: typing.SupportsInt) -> DiscOnMesh | None:
    ...