/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "tangentspace.hh"
#include "misc.hh"

#include <Eigen/Dense>

namespace AGPLib
{

TangentSpace::TangentSpace()
	: origin(Vec3d::Zero()), xAxis(Vec3d(1, 0, 0)), yAxis(Vec3d(0, 1, 0)) {}

TangentSpace::TangentSpace(const Vec3d& _xAxis, const Vec3d& _yAxis, const Vec3d& _origin)
	: origin(_origin), xAxis(_xAxis), yAxis(_yAxis) {}

TangentSpace::TangentSpace(const TangentSpace& _other)
	: origin(_other.origin), xAxis(_other.xAxis), yAxis(_other.yAxis) {}

Vec3d map_point_to_world_space(const Vec2d& p, const TangentSpace& tangent_space)
{
	return tangent_space.origin + p.x() * tangent_space.xAxis + p.y() * tangent_space.yAxis;
}

TangentSpace compute_any_tangent_space_basis(const Vec3d& normal, const Vec3d& origin)
{
	const Vec3d xAx = Util::any_orthogonal(normal);
	Vec3d yAx = normal.cross(xAx).normalized();

	return TangentSpace(xAx, yAx, origin);
}

}