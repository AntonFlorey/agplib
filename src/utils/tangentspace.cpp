/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include <Eigen/Dense>
#include "tangentspace.hh"

namespace
{

using namespace AGPLib;

Vec3d any_orthogonal(
	const Vec3d& _p)
{
	// Find coordinate axis spanning the largest angle with _p.
	// Return cross product that of axis with _p
	Vec3d tang;
	double min_abs_dot = INF_DOUBLE;
	for (const Vec3d& ax : { Vec3d(1.0, 0.0, 0.0), Vec3d(0.0, 1.0, 0.0), Vec3d(0.0, 0.0, 1.0) })
	{
		double abs_dot = std::abs(_p.dot(ax));
		if (abs_dot < min_abs_dot)
		{
			min_abs_dot = abs_dot;
			tang = ax.cross(_p).normalized();
		}
	}

	return tang;
}

}

namespace AGPLib
{

	TangentSpace::TangentSpace()
	{
		this->origin = Vec3d::Zero();
		this->xAxis = Vec3d(1, 0, 0);
		this->yAxis = Vec3d(0, 1, 0);
	}

	TangentSpace::TangentSpace(const Vec3d& _origin, const Vec3d& _xAxis, const Vec3d& _yAxis)
	{
		this->origin = _origin;
		this->xAxis = _xAxis;
		this->yAxis = _yAxis;
	}

	TangentSpace::TangentSpace(const Vec3d& _xAxis, const Vec3d& _yAxis)
	{
		this->origin = Vec3d::Zero();
		this->xAxis = _xAxis;
		this->yAxis = _yAxis;
	}

	TangentSpace::TangentSpace(const TangentSpace& _other)
	{
		this->origin = _other.origin;
		this->xAxis = _other.xAxis;
		this->yAxis = _other.yAxis;
	}

	Vec3d map_point_to_world_space(const Vec2d& p, const TangentSpace& tangent_space)
	{
		return tangent_space.origin + p.x() * tangent_space.xAxis + p.y() * tangent_space.yAxis;
	}

	TangentSpace compute_any_tangent_space_basis(const Vec3d& normal, const Vec3d& origin)
	{
		const Vec3d xAx = any_orthogonal(normal);
		Vec3d yAx = normal.cross(xAx).normalized();

		return TangentSpace(origin, xAx, yAx);
	}

}