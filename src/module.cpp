/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "module.hh"
#include "utils/curvature.hh"
#include "crossfield/surfacegraph.hh"

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int simple_test(const int n)
{
    std::cout << "Hello World" << std::endl;
    AGPLib::TangentSpace testSpace;
    AGPLib::VertexWithNormal testVertex(Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 0, 0));
    return 2 * n;
}

using namespace AGPLib;

PYBIND11_MODULE(agplib, m)
{
    m.doc() = "Anton's Geometry Processing Library: A small geometry processing package for the lazytopo Blender addon written in C++.";
        
    m.def("simple_test", &simple_test, "A simple test function that prints Hello World and multiplies an integer by two.", py::arg("n"));
    
    py::class_<VertexWithNormal>(m, "VertexWithNormal")
        .def(py::init<const Vec3d&, const Vec3d&>());

    py::class_<TangentSpace>(m, "TangentSpace")
        .def(py::init())
        .def(py::init<const Vec3d&, const Vec3d&>())
        .def(py::init<const Vec3d&, const Vec3d&, const Vec3d&>());
    
    py::class_<PrincipalCurvatureInfo>(m, "PrincipalCurvatureInfo")
        .def_readonly("direction", &PrincipalCurvatureInfo::direction)
        .def_readonly("unambiguity", &PrincipalCurvatureInfo::unambiguity);

    py::class_<SurfaceGraphNode>(m, "SurfaceGraphNode")
        .def(py::init<const double, const Vec3d&>());

    py::class_<SurfaceGraph>(m, "SurfaceGraph")
        .def(py::init())
        .def("add_node", &SurfaceGraph::add_node)
        .def("add_edge", &SurfaceGraph::add_edge)
        .def("number_of_nodes", &SurfaceGraph::number_of_nodes);

    m.def("approximate_II_fundamental_form", &approximate_II_fundamental_form, "Approximates the second fundamental form of a face with a least squares approach.");
    m.def("compute_principal_curvature", py::overload_cast<const std::vector<VertexWithNormal>&, const TangentSpace&>(&compute_principal_curvature),
        "computes the principal curvature direction and unambiguity score of a given face.");
    m.def("compute_principal_curvature", py::overload_cast<const std::vector<VertexWithNormal>&, const Vec3d&>(&compute_principal_curvature),
        "computes the principal curvature direction and unambiguity score of a given face.");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
