/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "types.hh"
#include "misc.hh"

#include <Eigen/Dense>

namespace AGPLib
{

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
		if (abs_dot < min_abs_dot && abs_dot < 1.0)
		{
			min_abs_dot = abs_dot;
			tang = ax.cross(_p).normalized();
		}
	}
	return tang;
}

}