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

#include <vector>
#include <unordered_set>

namespace AGPLib
{

struct SurfaceGraphNode
{
	double area;
	Vec3d normal;
	SurfaceGraphNode(const double _area, const Vec3d& _normal) : area(_area), normal(_normal) {};
};

struct SurfaceGraph
{
	using NodeID = size_t;

	std::vector<SurfaceGraphNode> nodes;
	std::vector<std::unordered_set<NodeID>> adj;

	SurfaceGraph() {};

	NodeID add_node(const SurfaceGraphNode& node);

	void add_edge(const NodeID idA, const NodeID idB);

	size_t number_of_nodes() const;
};

std::vector<SurfaceGraph::NodeID> create_meta_graph(
	const SurfaceGraph& orig_graph, 
	const std::vector<std::pair<SurfaceGraph::NodeID, SurfaceGraph::NodeID>>& merge_pairs,
	SurfaceGraph& meta_graph);

}