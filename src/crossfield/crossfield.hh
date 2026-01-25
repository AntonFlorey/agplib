/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 *
 * This crossfield implementation follows the approach introduced in "Instant Meshes":
 * https://igl.ethz.ch/projects/instant-meshes/
 *
 */

#pragma once

#include "types.hh"
#include "surfacegraph.hh"

#include <numbers>

namespace AGPLib
{
namespace Crossfield
{

struct CrossConstraint
{
	double weight;
	Vec3d direction;
	SurfaceGraph::NodeID sufaceNodeID;

	CrossConstraint(const double _weight, const Vec3d& _direction, const SurfaceGraph::NodeID _surfaceNodeID)
		: weight(_weight), direction(_direction), sufaceNodeID(_surfaceNodeID) {}
};

std::vector<Vec3d> compute_crossfield(
	const SurfaceGraph& surface,
	const std::vector<CrossConstraint>& constraints,
	const size_t max_iters = 20,
	const size_t max_multires_layers = 100,
	const double merge_normal_dot_th = 0.5,
	const double convergence_eps = 1e-6);

}
}