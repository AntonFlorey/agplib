/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "wrapperutils.hh"
#include "util/tangentspace.hh"
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
    memberBindings.emplace_back([agplib]() mutable
    {
        agplib
            .def("map_point_to_world_space", &map_point_to_world_space, py::arg("point"), py::arg("tangent_space"), CG_RELEASE)
            .def("compute_any_tangent_space_basis", &compute_any_tangent_space_basis, py::arg("normal"), py::arg("origin") = Vec3d::Zero(), CG_RELEASE);
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