/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#pragma once

#include "types.hh"

namespace AGPLib
{

struct TangentSpace
{
	Vec3d origin;
	Vec3d xAxis;
	Vec3d yAxis;

	TangentSpace();

	TangentSpace(const Vec3d& _origin, const Vec3d& _xAxis, const Vec3d& _yAxis);

	TangentSpace(const Vec3d& _xAxis, const Vec3d& _yAxis);

	TangentSpace(const TangentSpace& _other);
};

Vec3d map_point_to_world_space(const Vec2d& p, const TangentSpace& tangent_space);

TangentSpace compute_any_tangent_space_basis(const Vec3d& normal, const Vec3d& origin = Vec3d::Zero());

}