/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "wrapperutils.hh"
#include "util/tangentspace.hh"
#include "trimesh.hh"
#include "bindUtil.hh"
#include "bindPatchGraph.hh"
#include "bindCrossfield.hh"

#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace AGPLib;

namespace
{
void BindTopLevel(py::module& agplib, std::vector<std::function<void()>>& memberBindings)
{
    auto pyTangentSpace = py::class_<TangentSpace>(agplib, "TangentSpace");
    memberBindings.emplace_back([pyTangentSpace]() mutable
    {
        pyTangentSpace
            .def(py::init())
            .def(py::init<const Vec3d&, const Vec3d&>(), py::arg("x_axis"), py::arg("y_axis"))
            .def(py::init<const Vec3d&, const Vec3d&, const Vec3d&>(), py::arg("x_axis"), py::arg("y_axis"), py::arg("origin"));
    });
    auto pyTriMesh = py::class_<TriMesh>(agplib, "TriMesh");
    auto pyFaceSetData = py::class_<FaceSetData>(agplib, "FaceSetData");
    auto pyDiscOnMesh = py::class_<DiscOnMesh>(agplib, "DiscOnMesh");
    memberBindings.emplace_back([pyTriMesh, pyFaceSetData, pyDiscOnMesh]() mutable
    {
        pyTriMesh
            .def(py::init<
                const std::vector<Vec3d>&,
                const std::vector<Vec3d>&,
                const std::vector<VertexID>&,
                const std::vector<Vec3d>&,
                const std::vector<VertexID>&>(),
            py::arg("vertex_positions"),
            py::arg("vertex_normals"),
            py::arg("edge_vertices"),
            py::arg("face_normals"),
            py::arg("face_vertices"));
        pyFaceSetData
            .def_readonly("faces", &FaceSetData::faces)
            .def_readonly("edges", &FaceSetData::edges)
            .def_readonly("vertices", &FaceSetData::vertices);
        pyDiscOnMesh
            .def_readonly("face_set", &DiscOnMesh::face_set)
            .def_readonly("boundary_vertices", &DiscOnMesh::boundary_verts);
    });
    memberBindings.emplace_back([agplib]() mutable
    {
        agplib
            .def("map_point_to_world_space", &map_point_to_world_space, py::arg("point"), py::arg("tangent_space"), CG_RELEASE)
            .def("compute_any_tangent_space_basis", &compute_any_tangent_space_basis, py::arg("normal"), py::arg("origin") = Vec3d::Zero(), CG_RELEASE)
            .def("grow_face_subset_to_disc", &grow_face_subset_to_disc, py::arg("face_subset"), py::arg("mesh"), py::arg("max_grow_cycles"), CG_RELEASE)
            .def("flood_fill_face_subset_to_disc", &flood_fill_face_subset_to_disc, py::arg("face_subset"), py::arg("mesh"), py::arg("start_face"), CG_RELEASE);
    });
}
}

PYBIND11_MODULE(_agplib, agplib)
{
    using namespace AGPLib::PythonWrapper;

    std::vector<std::function<void()>> memberBindings;

    agplib.doc() = "A(nton's) Geometry Processing Library: A small geometry processing package for the lazytopo Blender addon written in C++.";

    BindTopLevel(agplib, memberBindings);
    BindUtil(agplib, memberBindings);
    BindCrossfield(agplib, memberBindings);
    BindPatchGraph(agplib, memberBindings);

    for (auto &f : memberBindings)
    {
        f();
    }

#ifdef VERSION_INFO
    agplib.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    agplib.attr("__version__") = "dev";
#endif

}