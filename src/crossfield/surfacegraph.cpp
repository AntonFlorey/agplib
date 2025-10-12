/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 * 
 * This implementation follows the approach introduced in "Instant Meshes":
 * https://igl.ethz.ch/projects/instant-meshes/
 * 
 */

#include "surfacegraph.hh"

#include <TinyAD/Utils/Out.hh>

namespace AGPLib
{

SurfaceGraph::NodeID SurfaceGraph::add_node(const SurfaceGraphNode& node)
{
	nodes.push_back(node);
	adj.push_back(std::unordered_set<NodeID>());
	return nodes.size() - 1;
}

void SurfaceGraph::add_edge(const NodeID idA, const NodeID idB)
{
	if (idA >= nodes.size() || idB >= nodes.size())
	{
		TINYAD_ERROR("Adding an edge to surface graph failed. Both node ids must be valid!");
		return;
	}
	adj[idA].insert(idB);
	adj[idB].insert(idA);
}

size_t SurfaceGraph::number_of_nodes() const
{
	return nodes.size();
}

std::vector<SurfaceGraph::NodeID> create_meta_graph(
	const SurfaceGraph& orig_graph,
	const std::vector<std::pair<SurfaceGraph::NodeID, SurfaceGraph::NodeID>>& merge_pairs,
	SurfaceGraph& meta_graph)
{
	size_t num_meta_nodes = orig_graph.number_of_nodes() - merge_pairs.size();
	meta_graph.nodes.clear();
	meta_graph.nodes.reserve(num_meta_nodes);
	meta_graph.adj.clear();
	meta_graph.adj.reserve(num_meta_nodes);

	// Create all meta nodes
	std::vector<bool> orig_node_has_beeen_merged(orig_graph.number_of_nodes(), false);
	std::vector<SurfaceGraph::NodeID> orig_node_to_meta_node_map(orig_graph.number_of_nodes(), 0);
	for (const std::pair<SurfaceGraph::NodeID, SurfaceGraph::NodeID>& merge_pair : merge_pairs)
	{
		const SurfaceGraphNode first_node = orig_graph.nodes[merge_pair.first];
		const SurfaceGraphNode second_node = orig_graph.nodes[merge_pair.second];
		double meta_area = first_node.area + second_node.area;
		Vec3d meta_normal = (first_node.normal + second_node.normal).normalized();
		SurfaceGraph::NodeID meta_node_id = meta_graph.add_node(SurfaceGraphNode(meta_area, meta_normal));
		orig_node_has_beeen_merged[merge_pair.first] = true;
		orig_node_has_beeen_merged[merge_pair.second] = true;
		orig_node_to_meta_node_map[merge_pair.first] = meta_node_id;
		orig_node_to_meta_node_map[merge_pair.second] = meta_node_id;
	}
	// Copy non-merged nodes
	for (SurfaceGraph::NodeID orig_node_id = 0; orig_node_id < orig_graph.number_of_nodes(); orig_node_id++)
	{
		if (orig_node_has_beeen_merged[orig_node_id])
		{
			continue;
		}
		SurfaceGraph::NodeID meta_node_id = meta_graph.add_node(orig_graph.nodes[orig_node_id]);
		orig_node_to_meta_node_map[orig_node_id] = meta_node_id;
	}
	// Create all meta-edges
	for (SurfaceGraph::NodeID orig_node_id = 0; orig_node_id < orig_graph.number_of_nodes(); orig_node_id++)
	{
		SurfaceGraph::NodeID meta_node_id = orig_node_to_meta_node_map[orig_node_id];
		for (SurfaceGraph::NodeID orig_nb_id : orig_graph.adj[orig_node_id])
		{
			SurfaceGraph::NodeID meta_nb_id = orig_node_to_meta_node_map[orig_nb_id];
			if (meta_node_id >= meta_nb_id) // Every pair will appear in both possible orders
			{
				continue;
			}
			meta_graph.add_edge(meta_node_id, meta_nb_id);
		}
	}
	return orig_node_to_meta_node_map;
}

}