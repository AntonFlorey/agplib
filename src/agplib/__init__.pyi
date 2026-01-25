"""
A(nton's) Geometry Processing Library: A small geometry processing package for the lazytopo Blender addon written in C++.
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
from . import crossfield
from . import util

__all__ = [
    'TangentSpace', 
    'compute_any_tangent_space_basis', 
    'crossfield', 
    'map_point_to_world_space', 
    'util'
    ]

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
__version__: str = ...
