from __future__ import annotations
import numpy
import numpy.typing
import typing
import collections.abc

__all__ = ['any_orthogonal', 'make_orthogonal_dir', 'compute_convex_hull']

def any_orthogonal(point: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    ...
def make_orthogonal_dir(vector: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"], 
                        normal: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    ...
def compute_convex_hull(points: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]], eps: typing.SupportsFloat = 1e-12) -> list[int]:
    """Computes the convex hull of a 2D point cloud. Returns the indices of hull vertices in ccw order."""
    ...
