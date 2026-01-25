/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#pragma once

#include "types.hh"
#include "util/tangentspace.hh"

namespace AGPLib
{
namespace Crossfield
{

struct VertexWithNormal
{
	Vec3d pos;
	Vec3d normal;

	VertexWithNormal(const Vec3d& _pos, const Vec3d& _normal);
};

struct PrincipalCurvatureInfo
{
	Vec3d direction;
	double unambiguity;

	PrincipalCurvatureInfo();

	PrincipalCurvatureInfo(const Vec3d& dir, const double unambiguity);

	PrincipalCurvatureInfo(const PrincipalCurvatureInfo& _other);
};

Mat2d approximate_II_fundamental_form(const std::vector<VertexWithNormal>& face_verts, const TangentSpace& tangent_space);

PrincipalCurvatureInfo compute_principal_curvature(const std::vector<VertexWithNormal>& face_verts, const TangentSpace& tangent_space);

PrincipalCurvatureInfo compute_principal_curvature(const std::vector<VertexWithNormal>& face_verts, const Vec3d& face_normal);

}
}