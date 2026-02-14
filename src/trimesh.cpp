/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "trimesh.hh"
#include "util/printhelpers.hh"
#include "TinyAD/Utils/Out.hh"

#include <unordered_map>

namespace AGPLib
{

namespace
{

FaceSetData get_all_face_set_data(const std::unordered_set<FaceID>& face_set, const TriMesh& mesh)
{
    FaceSetData res;
    res.faces = face_set;
    res.vertices.clear();
    res.edges.clear();
    for (FaceID f_id : face_set)
    {
        for (EdgeID e_id : mesh.face_edges[f_id])
        {
            res.edges.insert(e_id);
        }
        for (VertexID v_id : mesh.face_vertices[f_id])
        {
            res.vertices.insert(v_id);
        }
    }
    return res;
}

bool edge_is_boundary_of_face_set(EdgeID edge, const FaceSetData& face_set, const TriMesh& mesh)
{
    if (!face_set.edges.contains(edge)) return false;
    if (mesh.is_boundary_edge(edge)) return true;
    return !(face_set.faces.contains(mesh.edge_faces[edge][0]) && face_set.faces.contains(mesh.edge_faces[edge][1]));
}

std::vector<std::vector<VertexID>> get_face_set_boundaries(const FaceSetData& face_set, const TriMesh& mesh)
{
    std::unordered_set<EdgeID> visited_edges;
    const auto check_if_visited = [&visited_edges](EdgeID e_id){
        if (visited_edges.contains(e_id)) return true;
        visited_edges.insert(e_id);
        return false;
    };
    std::vector<std::vector<VertexID>> boundaries;

    for (EdgeID e_id : face_set.edges)
    {
        if (check_if_visited(e_id)) continue;
        if (!edge_is_boundary_of_face_set(e_id, face_set, mesh)) continue;
        // We have a found a new boundary
        std::vector<VertexID> boundary_vertices;
        EdgeID current_boundary_edge = e_id;
        const VertexID start_vertex = mesh.edge_vertices[current_boundary_edge].v_id_A;
        VertexID current_vertex = mesh.edge_vertices[current_boundary_edge].v_id_B;
        boundary_vertices.push_back(start_vertex);
        size_t search_couter = 0;
        while (current_vertex != start_vertex && search_couter <= face_set.edges.size())
        {
            boundary_vertices.push_back(current_vertex);
            std::optional<EdgeID> next_boundary_edge = std::nullopt;
            for (EdgeID adj_edge : mesh.vertex_edge_rings[current_vertex])
            {
                if (check_if_visited(adj_edge)) 
                {
                    continue;
                }
                if (adj_edge == current_boundary_edge)
                { 
                    continue;
                }
                if (edge_is_boundary_of_face_set(adj_edge, face_set, mesh))
                { 
                    next_boundary_edge = adj_edge;
                    break;
                }
            }
            TINYAD_ASSERT(next_boundary_edge.has_value());
            current_vertex = mesh.get_other_vertex(current_vertex, next_boundary_edge.value());
            current_boundary_edge = next_boundary_edge.value();
            search_couter++;
        }
        TINYAD_ASSERT(current_vertex == start_vertex);
        boundaries.push_back(boundary_vertices);
    }
    return boundaries;
}

bool face_set_has_disc_topology(const FaceSetData& face_set, const TriMesh& mesh)
{   
    // Check euler characteristic
    const long n_verts = static_cast<long>(face_set.vertices.size());
    const long n_edges = static_cast<long>(face_set.edges.size());
    const long n_faces = static_cast<long>(face_set.faces.size());
    const long euler_chi = n_verts - n_edges + n_faces;
    TINYAD_DEBUG_OUT("Euler characteristic is: " + std::to_string(euler_chi));
    if (euler_chi != 1) return false;

    // Check for non-manifold vertices (outgoing boundary edges not in {0,2})
    for (VertexID v_id : face_set.vertices)
    {
        int current_number_of_boundary_edges = 0;
        for (EdgeID e_id : mesh.vertex_edge_rings[v_id])
        {
            if (edge_is_boundary_of_face_set(e_id, face_set, mesh)) current_number_of_boundary_edges ++;
            if (current_number_of_boundary_edges > 2) return false;
        }
        if (current_number_of_boundary_edges == 1) return false;
    }

    // Then check for exactly one boundary
    const auto boundaries = get_face_set_boundaries(face_set, mesh);
    return boundaries.size() == 1;
}

bool face_set_has_disc_topology(const std::unordered_set<FaceID>& face_set, const TriMesh& mesh)
{
    TINYAD_DEBUG_OUT("Checking this set for disc topology: " + Util::id_container_to_string(face_set));
    return face_set_has_disc_topology(get_all_face_set_data(face_set, mesh), mesh);
}

}// namespace

TriMesh::TriMesh(
    const std::vector<Vec3d>& _vertex_positions,
    const std::vector<Vec3d>& _vertex_normals,
    const std::vector<VertexID>& _edge_vertices,
    const std::vector<Vec3d>& _face_normals,
    const std::vector<VertexID>& _face_vertices)
    : 
    n_vertices(_vertex_positions.size()), 
    n_edges(_edge_vertices.size() / 2),
    n_faces(_face_normals.size())
{
    vertex_positions = _vertex_positions;
    vertex_normals = _vertex_normals;
    face_normals = _face_normals;

    std::unordered_map<EdgeKey, EdgeID> id_of_edge_key;
    edge_vertices = std::vector<EdgeKey>();
    edge_vertices.reserve(n_edges);
    vertex_vertex_rings = std::vector<std::vector<VertexID>>(n_vertices, std::vector<VertexID>());
    vertex_edge_rings = std::vector<std::vector<EdgeID>>(n_vertices, std::vector<EdgeID>());
    for (EdgeID e_id = 0; e_id < n_edges; e_id++)
    {   
        VertexID v_A = _edge_vertices[e_id * 2];
        VertexID v_B = _edge_vertices[(e_id * 2) + 1];
        EdgeKey current_key(v_A, v_B);
        edge_vertices.push_back(current_key);
        id_of_edge_key[current_key] = e_id;
        vertex_vertex_rings[v_A].push_back(v_B);
        vertex_vertex_rings[v_B].push_back(v_A);
        vertex_edge_rings[v_A].push_back(e_id);
        vertex_edge_rings[v_B].push_back(e_id);
    }

    face_vertices = std::vector<std::array<VertexID, 3>>();
    face_vertices.reserve(n_faces);
    face_edges = std::vector<std::array<EdgeID, 3>>();
    face_edges.reserve(n_faces);
    edge_faces = std::vector<std::vector<FaceID>>(n_edges, std::vector<FaceID>());
    for (FaceID f_id = 0; f_id < n_faces; f_id++)
    {
        VertexID v_A = _face_vertices[f_id * 3];
        VertexID v_B = _face_vertices[(f_id * 3) + 1];
        VertexID v_C = _face_vertices[(f_id * 3) + 2];
        EdgeKey edge_AB(v_A, v_B);
        EdgeKey edge_BC(v_B, v_C);
        EdgeKey edge_CA(v_C, v_A);
        face_vertices.push_back({v_A, v_B, v_C});
        edge_faces[id_of_edge_key[edge_AB]].push_back(f_id);
        edge_faces[id_of_edge_key[edge_BC]].push_back(f_id);
        edge_faces[id_of_edge_key[edge_CA]].push_back(f_id);
        face_edges.push_back({id_of_edge_key[edge_AB], id_of_edge_key[edge_BC], id_of_edge_key[edge_CA]});
    }

    face_nbs = std::vector<std::vector<FaceID>>(n_faces, std::vector<FaceID>());
    for (EdgeID e_id = 0; e_id < n_edges; e_id++)
    {
        if (is_boundary_edge(e_id)) continue;
        FaceID f_A = edge_faces[e_id][0];
        FaceID f_B = edge_faces[e_id][1];
        face_nbs[f_A].push_back(f_B);
        face_nbs[f_B].push_back(f_A);
    }
}

bool TriMesh::is_boundary_edge(EdgeID e_id) const
{
    return !(edge_faces[e_id].size() == 2);
}

std::optional<FaceID> TriMesh::get_opposite_face(EdgeID e_id, FaceID f_id) const
{
    if (is_boundary_edge(e_id)) return std::nullopt;
    return edge_faces[e_id][0] == f_id ? edge_faces[e_id][1] : edge_faces[e_id][0];
}

VertexID TriMesh::get_other_vertex(VertexID v_id, EdgeID e_id) const
{
    const VertexID v_A = edge_vertices[e_id].v_id_A;
    const VertexID v_B = edge_vertices[e_id].v_id_B;
    if (v_A == v_id)
    {
        return v_B;
    }
    else if (v_B == v_id)
    {
        return v_A;
    }
    else
    {
        return v_id;
    }
}

std::optional<DiscOnMesh> grow_face_subset_to_disc(
    const std::vector<FaceID>& face_subset, 
    const TriMesh& mesh,
    int max_grow_cycles)
{
    std::unordered_set<FaceID> current_face_set(face_subset.begin(), face_subset.end());
    bool current_set_is_disc = face_set_has_disc_topology(current_face_set, mesh);

    TINYAD_INFO("Growing disc starting with faces: " + Util::id_container_to_string(current_face_set));
    TINYAD_INFO("Start faces are disc: " + std::to_string(current_set_is_disc));

    std::unordered_set<FaceID> added_last_round = current_face_set;
    for (int grow_cycle = 0; grow_cycle < max_grow_cycles; grow_cycle++)
    {
        // collect next face front
        std::unordered_set<FaceID> next_front;
        for (FaceID face_last_added : added_last_round)
        {
            for (FaceID nb_face : mesh.face_nbs[face_last_added])
            {
                if (current_face_set.contains(nb_face)) continue;
                next_front.insert(nb_face);
            }
        }
        current_face_set.insert(next_front.begin(), next_front.end());
        added_last_round = next_front;
        const bool next_set_is_disc = face_set_has_disc_topology(current_face_set, mesh);

        TINYAD_DEBUG_OUT("Face set grew to: " + Util::id_container_to_string(current_face_set));
        TINYAD_DEBUG_OUT("Is it now a disc? " + std::to_string(next_set_is_disc));

        if (current_set_is_disc && !next_set_is_disc)
        {
            // roll one step back and break early
            TINYAD_DEBUG_OUT("Rolling back the face subset!");
            current_face_set.erase(next_front.begin(), next_front.end()); // this works since sets are disjoint
            break;
        }
        current_set_is_disc = next_set_is_disc;
    }
    if (!current_set_is_disc) return std::nullopt;
    TINYAD_DEBUG_OUT("Returning face set: " + Util::id_container_to_string(current_face_set));
    const auto full_disc_data = get_all_face_set_data(current_face_set, mesh);
    const auto boundaries = get_face_set_boundaries(full_disc_data, mesh);
    return DiscOnMesh(full_disc_data, boundaries[0]);
}

std::optional<DiscOnMesh> flood_fill_face_subset_to_disc(
    const std::vector<FaceID>& face_subset,
    const TriMesh& mesh,
    FaceID& start_face)
{
    return std::nullopt; // TODO
} 

}