/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */


#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "curvature.hh"

namespace AGPLib
{
namespace Crossfield
{

VertexWithNormal::VertexWithNormal(const Vec3d& _pos, const Vec3d& _normal)
{
	this->pos = _pos;
	this->normal = _normal;
}

PrincipalCurvatureInfo::PrincipalCurvatureInfo()
{
	this->direction = Vec3d::Zero();
	this->unambiguity = 0.0;
}

PrincipalCurvatureInfo::PrincipalCurvatureInfo(const Vec3d& dir, const double unambiguity)
{
	this->direction = dir;
	this->unambiguity = unambiguity;
}

PrincipalCurvatureInfo::PrincipalCurvatureInfo(const PrincipalCurvatureInfo& _other)
{
	this->direction = _other.direction;
	this->unambiguity = _other.unambiguity;
}

Mat2d approximate_II_fundamental_form(const std::vector<VertexWithNormal>& face_verts, const TangentSpace& tangent_space)
{
	int num_verts = face_verts.size();
	int eq_rows = 2 * num_verts;
	MatXd A = MatXd::Zero(eq_rows, 3);
	VecXd rhs = VecXd::Zero(eq_rows);

	for (int i = 0; i < num_verts; i++)
	{
		int j = (i + 1) % num_verts;
		int mat_row = 2 * i;

		Vec3d delta_pos = face_verts[j].pos - face_verts[i].pos;
		Vec3d delta_normal = face_verts[j].normal - face_verts[i].normal;

		A.row(mat_row) << delta_pos.dot(tangent_space.xAxis), delta_pos.dot(tangent_space.yAxis), 0.0;
		A.row(mat_row + 1) << 0.0, delta_pos.dot(tangent_space.xAxis), delta_pos.dot(tangent_space.yAxis);
		rhs[mat_row] = delta_normal.dot(tangent_space.xAxis);
		rhs[mat_row + 1] = delta_normal.dot(tangent_space.yAxis);
	}

	VecXd lstsq_sol = A.colPivHouseholderQr().solve(rhs);
	Mat2d II;
	II << lstsq_sol[0], lstsq_sol[1], lstsq_sol[1], lstsq_sol[2];
	return II;
}

PrincipalCurvatureInfo compute_principal_curvature(const std::vector<VertexWithNormal>& face_verts, const TangentSpace& tangent_space)
{
	Mat2d II = approximate_II_fundamental_form(face_verts, tangent_space);

	Eigen::SelfAdjointEigenSolver<Mat2d> solver;
	solver.compute(II);
	Vec2d eigvals = solver.eigenvalues();
	Vec2d min_curvature_dir_2d = solver.eigenvectors().col(0);

	PrincipalCurvatureInfo info(map_point_to_world_space(min_curvature_dir_2d, tangent_space), std::abs(eigvals.x() - eigvals.y()));
	return info;
}

PrincipalCurvatureInfo compute_principal_curvature(const std::vector<VertexWithNormal>& face_verts, const Vec3d& face_normal)
{
	TangentSpace local_tangent_space = compute_any_tangent_space_basis(face_normal);
	return compute_principal_curvature(face_verts, local_tangent_space);
}

}
}