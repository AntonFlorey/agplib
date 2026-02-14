/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#pragma once

#include "types.hh"
#include "util/misc.hh"

#include <optional>
#include <unordered_set>

namespace AGPLib
{

using VertexID = size_t;
using EdgeID = size_t;
using FaceID = size_t;

struct EdgeKey
{
    size_t v_id_A;
    size_t v_id_B;

    EdgeKey(size_t _v_id_A, size_t _v_id_B)
    {
        v_id_A = _v_id_A;
        v_id_B = _v_id_B;
        if (_v_id_A > _v_id_B)
        {
            std::swap(v_id_A, v_id_B);    
        }
    }

    bool operator == (const EdgeKey& other) const
    {
        return this->v_id_A == other.v_id_A && this->v_id_B == other.v_id_B;
    }
};

struct TriMesh
// Rudimentary triangle mesh struct
{
    // Vertex data
    size_t n_vertices;
    std::vector<Vec3d> vertex_positions;
    std::vector<Vec3d> vertex_normals;
    std::vector<std::vector<EdgeID>> vertex_edge_rings;
    std::vector<std::vector<VertexID>> vertex_vertex_rings;

    // Edge data
    size_t n_edges;
    std::vector<EdgeKey> edge_vertices;
    std::vector<std::vector<FaceID>> edge_faces;

    // Face data
    size_t n_faces;
    std::vector<Vec3d> face_normals;
    std::vector<std::array<VertexID, 3>> face_vertices;
    std::vector<std::array<EdgeID, 3>> face_edges;
    std::vector<std::vector<FaceID>> face_nbs;

    TriMesh(const std::vector<Vec3d>& _vertex_positions,
            const std::vector<Vec3d>& _vertex_normals,
            const std::vector<VertexID>& _edge_vertices,
            const std::vector<Vec3d>& _face_normals,
            const std::vector<VertexID>& _face_vertices);

    bool is_boundary_edge(EdgeID e_id) const ;

    std::optional<FaceID> get_opposite_face(EdgeID e_id, FaceID f_id) const;

    VertexID get_other_vertex(VertexID v_id, EdgeID e_id) const;
};

struct FaceSetData
{
    std::unordered_set<FaceID> faces;
    std::unordered_set<VertexID> vertices;
    std::unordered_set<EdgeID> edges;
};

struct DiscOnMesh
{
    FaceSetData face_set;
    std::vector<VertexID> boundary_verts;
};

std::optional<DiscOnMesh> grow_face_subset_to_disc(
    const std::vector<FaceID>& face_subset, 
    const TriMesh& mesh,
    int max_grow_cycles);

std::optional<DiscOnMesh> flood_fill_face_subset_to_disc(
    const std::vector<FaceID>& face_subset,
    const TriMesh& mesh,
    FaceID& start_face);    


}

namespace std
{

template<>
struct hash<AGPLib::EdgeKey>
{
	size_t operator()(const AGPLib::EdgeKey& key) const
	{
		size_t hash = 0;
        AGPLib::Util::hash_combine(hash, key.v_id_A);
        AGPLib::Util::hash_combine(hash, key.v_id_B);
		return hash;
	}
};

}