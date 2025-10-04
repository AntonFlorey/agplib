#include <iostream>
#include <pybind11/pybind11.h>

#include "module.hh"

namespace py = pybind11;

int simple_test(const int n)
{
    std::cout << "Hello World" << std::endl;
    return 2 * n;
}

PYBIND11_MODULE(agplib, m)
{
    m.doc() = "Anton's Geometry Processing Library: A small geometry processing package for the lazytopo Blender addon written in C++.";
        
    m.def("simple_test", &simple_test, "A simple test function that prints Hello World and multiplies an integer by two.", py::arg("n"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
