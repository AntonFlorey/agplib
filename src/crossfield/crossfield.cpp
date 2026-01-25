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

#include "crossfield.hh"
#include "util/misc.hh"

#include <algorithm>
#include <Eigen/Dense>

#include "TinyAD/Utils/Out.hh"

namespace
{

using namespace AGPLib;
using namespace AGPLib::Crossfield;

using NodeToContraintMap = std::vector<std::vector<CrossConstraint>>;
using MergePair = std::pair<SurfaceGraph::NodeID, SurfaceGraph::NodeID>;
using MergeCandidate = std::pair<double, MergePair>;

NodeToContraintMap compute_surface_node_to_constraint_map(const SurfaceGraph& surface, const std::vector<CrossConstraint>& constraints)
{
	NodeToContraintMap map(surface.number_of_nodes(), std::vector<CrossConstraint>());
	for (const CrossConstraint& constraint : constraints)
	{
		map[constraint.sufaceNodeID].push_back(constraint);
	}
	return map;
}

bool collect_merge_candidates(const SurfaceGraph& surface, const double merge_normal_dot_th, std::vector<MergeCandidate>& candidates)
{
	for (SurfaceGraph::NodeID node_id = 0; node_id < surface.number_of_nodes(); node_id++)
	{
		double node_area = surface.nodes[node_id].area;
		Vec3d node_normal = surface.nodes[node_id].normal;
		for (SurfaceGraph::NodeID nb_id : surface.adj[node_id])
		{
			if (nb_id < node_id)
			{
				continue;
			}
			double nb_area = surface.nodes[nb_id].area;
			Vec3d nb_normal = surface.nodes[nb_id].normal;
			double normal_dot = node_normal.dot(nb_normal);
			if (normal_dot < merge_normal_dot_th)
			{
				continue;
			}
			double area_ratio = node_area > nb_area ? (node_area / nb_area) : (nb_area / node_area);
			candidates.push_back({ normal_dot * area_ratio, {node_id, nb_id} });
		}
	}
	return candidates.size() != 0;
}

std::pair<Vec3d,Vec3d> align_crosses(const Vec3d& crossA, const Vec3d& normalA, const Vec3d& crossB, const Vec3d& normalB)
{	
	double best_score = -1.0;
	Vec3d alignedA;
	Vec3d alignedB;
	std::array<Vec3d, 2> optionsA = { crossA, normalA.cross(crossA) };
	std::array<Vec3d, 2> optionsB = { crossB, normalB.cross(crossB) };
	for (const Vec3d& optionA : optionsA)
	{
		for (const Vec3d& optionB : optionsB)
		{
			double score = std::abs(optionA.dot(optionB));
			if (score > best_score)
			{
				best_score = score;
				alignedA = optionA;
				alignedB = optionB;
			}
		}
	}
	double signB = alignedA.dot(alignedB) >= 0 ? 1.0 : -1.0;
	return std::pair<Vec3d, Vec3d>(alignedA, signB * alignedB);
}

}

namespace AGPLib
{
namespace Crossfield
{
std::vector<Vec3d> compute_crossfield(
	const SurfaceGraph& surface,
	const std::vector<CrossConstraint>& constraints,
	const size_t max_iters,
	const size_t max_multires_layers,
	const double merge_normal_dot_th,
	const double convergence_eps)
{
	std::vector<Vec3d> crossfield;
	if (max_multires_layers > 0)
	{
		std::vector<MergeCandidate> merge_candidates;
		if (collect_merge_candidates(surface, merge_normal_dot_th, merge_candidates))
		{
			std::sort(merge_candidates.begin(), merge_candidates.end(), [](const MergeCandidate& a, const MergeCandidate& b)
			{
				return a.first > b.first;
			});
			std::vector<MergePair> sorted_merge_pairs;
			for (const MergeCandidate& merge_candidate : merge_candidates)
			{
				sorted_merge_pairs.push_back(merge_candidate.second);
			}
			SurfaceGraph meta_surface;
			std::vector<SurfaceGraph::NodeID> orig_node_to_meta_node_map = create_meta_graph(surface, sorted_merge_pairs, meta_surface);

			// transfer constraints to meta surface
			std::vector<CrossConstraint> meta_constraints;
			for (const CrossConstraint& orig_constraint : constraints)
			{
				SurfaceGraph::NodeID orig_node_id = orig_constraint.sufaceNodeID;
				SurfaceGraph::NodeID meta_node_id = orig_node_to_meta_node_map[orig_constraint.sufaceNodeID];
				double orig_node_area = surface.nodes[orig_node_id].area;
				double meta_node_area = meta_surface.nodes[meta_node_id].area;
				CrossConstraint meta_constraint(
					orig_node_area * orig_constraint.weight / meta_node_area,
					Util::make_orthogonal_dir(orig_constraint.direction, meta_surface.nodes[meta_node_id].normal),
					meta_node_id);
				meta_constraints.push_back(meta_constraint);
			}
			// Revursively compute crossfield of meta surface
			std::vector<Vec3d> meta_crossfield = compute_crossfield(
				meta_surface,
				meta_constraints,
				max_iters,
				max_multires_layers - 1,
				merge_normal_dot_th,
				convergence_eps
			);
			// Transfer the computed meta-crossfield to this surface
			for (SurfaceGraph::NodeID orig_node_id = 0; orig_node_id < surface.number_of_nodes(); orig_node_id++)
			{
				SurfaceGraph::NodeID meta_node_id = orig_node_to_meta_node_map[orig_node_id];
				crossfield.push_back(Util::make_orthogonal_dir(meta_crossfield[meta_node_id], meta_surface.nodes[meta_node_id].normal));
			}
		}
	}
	
	if (crossfield.size() == 0)
	{
		// Initialize with random directions in tangent space
		for (SurfaceGraph::NodeID node_id = 0; node_id < surface.number_of_nodes(); node_id++)
		{
			crossfield.push_back(Util::any_orthogonal(surface.nodes[node_id].normal));
		}
	}

	// Optimize the crossfield
	NodeToContraintMap constraint_map = compute_surface_node_to_constraint_map(surface, constraints);
	for (size_t optimization_iteration = 0; optimization_iteration < max_iters; optimization_iteration++)
	{
		double max_change = 0;
		for (SurfaceGraph::NodeID node_id = 0; node_id < surface.number_of_nodes(); node_id++)
		{
			SurfaceGraphNode node = surface.nodes[node_id];
			double weight_sum = 0;
			Vec3d new_crossdir = crossfield[node_id]; // just in case this is an isolated area

			auto collect_crossdir = [&new_crossdir, &node, &weight_sum](const Vec3d& crossdir, const double weight, const Vec3d& normal)
			{
				std::pair<Vec3d, Vec3d> aligned_crosses = align_crosses(new_crossdir, node.normal, crossdir, normal);
				new_crossdir = weight_sum * aligned_crosses.first + weight * aligned_crosses.second;
				new_crossdir -= node.normal * node.normal.dot(new_crossdir);
				new_crossdir.normalize();
				weight_sum += weight;
			};

			for (SurfaceGraph::NodeID nb_id : surface.adj[node_id])
			{
				collect_crossdir(crossfield[nb_id], 1.0, surface.nodes[nb_id].normal);
			}
			for (const CrossConstraint& constraint : constraint_map[node_id])
			{
				collect_crossdir(constraint.direction, constraint.weight, node.normal);
			}
			std::pair<Vec3d, Vec3d> aligned_old_new = align_crosses(crossfield[node_id], node.normal, new_crossdir, node.normal);
			max_change = std::max(max_change, (aligned_old_new.first - aligned_old_new.second).norm());
			crossfield[node_id] = new_crossdir;
		}
		// Convergence check
		if (max_change < convergence_eps)
		{
			break;
		}
	}
	return crossfield;
}

}
}