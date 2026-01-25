/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#pragma once

#include "types.hh"

namespace AGPLib
{
namespace mathutil
{

bool is_near_zero(double x, double eps = EPS) 
{
    return std::abs(x) <= eps;
}

bool is_equal(double x, double y, double eps = EPS)
{
    return is_near_zero(x - y, eps);
}

bool is_clearly_greater(double x, double y, double eps = EPS)
{
    return x > y + eps;
}


}
}