/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "bindCrossfield.hh"

#include "crossfield/crossfield.hh"
#include "crossfield/curvature.hh"
#include "crossfield/surfacegraph.hh"

namespace py = pybind11;

namespace AGPLib
{
namespace PythonWrapper
{

void BindCrossfield(py::module& agplib, std::vector<std::function<void()>>& memberBindings)
{
	using namespace Crossfield;

	auto crossfield_submodule = agplib.def_submodule("crossfield");

    auto pyVertexWithNormal = py::class_<VertexWithNormal>(crossfield_submodule, "VertexWithNormal");
    auto pySurfaceGraphNode = py::class_<SurfaceGraphNode>(crossfield_submodule, "SurfaceGraphNode");
    auto pySurfaceGraph = py::class_<SurfaceGraph>(crossfield_submodule, "SurfaceGraph");
    memberBindings.emplace_back([pyVertexWithNormal, pySurfaceGraphNode, pySurfaceGraph]() mutable
    {
        pyVertexWithNormal.def(py::init<const Vec3d&, const Vec3d&>(), py::arg("pos"), py::arg("normal"));
        pySurfaceGraphNode.def(py::init<const double, const Vec3d&>(), py::arg("area"), py::arg("normal"));
        pySurfaceGraph
            .def(py::init())
            .def("add_node", &SurfaceGraph::add_node, py::arg("node"), CG_RELEASE)
            .def("add_edge", &SurfaceGraph::add_edge, py::arg("node_id_A"), py::arg("node_id_B"), CG_RELEASE)
            .def("number_of_nodes", &SurfaceGraph::number_of_nodes, CG_RELEASE);
    });

    auto pyPrincipalCurvatureInfo = py::class_<PrincipalCurvatureInfo>(crossfield_submodule, "PrincipalCurvatureInfo");
    auto pyCrossConstraint = py::class_<CrossConstraint>(crossfield_submodule, "CrossConstraint");
    memberBindings.emplace_back([pyPrincipalCurvatureInfo, pyCrossConstraint]() mutable
    {
        pyPrincipalCurvatureInfo
            .def_readonly("direction", &PrincipalCurvatureInfo::direction)
            .def_readonly("unambiguity", &PrincipalCurvatureInfo::unambiguity);
        pyCrossConstraint.def(py::init<const double, const Vec3d&, const SurfaceGraph::NodeID>(), py::arg("weight"), py::arg("direction"), py::arg("surface_node_id"));
    });

	memberBindings.emplace_back([crossfield_submodule]() mutable
	{
        crossfield_submodule
            .def("approximate_II_fundamental_form", &approximate_II_fundamental_form, CG_RELEASE)
            .def("compute_principal_curvature", 
                py::overload_cast<const std::vector<VertexWithNormal>&, 
                const TangentSpace&>(&compute_principal_curvature), 
                py::arg("face_verts"),
                py::arg("tangent_space"),
                CG_RELEASE)
            .def("compute_principal_curvature", 
                py::overload_cast<const std::vector<VertexWithNormal>&, 
                const Vec3d&>(&compute_principal_curvature), 
                py::arg("face_verts"),
                py::arg("face_normal"), 
                CG_RELEASE)
            .def("compute_crossfield", &compute_crossfield,
                py::arg("surface"),
                py::arg("constraints"),
                py::arg("max_iters") = 20,
                py::arg("max_multires_layers") = 100,
                py::arg("merge_normal_dot_th") = 0.5,
                py::arg("convergence_eps") = 1e-6, 
                CG_RELEASE);
	});
}

}
}
