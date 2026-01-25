/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "bindUtil.hh"

#include "util/misc.hh"
#include "util/convexhull.hh"

namespace py = pybind11;

namespace AGPLib
{
namespace PythonWrapper
{

void BindUtil(py::module& agplib, std::vector<std::function<void()>>& memberBindings)
{
	using namespace Util;

	auto util_submodule = agplib.def_submodule("util");

	memberBindings.emplace_back([util_submodule]() mutable
	{
		util_submodule
			.def("any_orthogonal", &any_orthogonal, py::arg("point"), CG_RELEASE)
			.def("make_orthogonal_dir", &make_orthogonal_dir, py::arg("vector"), py::arg("normal"), CG_RELEASE)
			.def("compute_convex_hull", &compute_convex_hull, py::arg("points"), py::arg("eps") = EPS, CG_RELEASE);
	});
}

}
}
