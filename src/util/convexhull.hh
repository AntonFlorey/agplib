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
namespace Util
{

std::vector<size_t> compute_convex_hull(const std::vector<Vec2d>& points, double eps = EPS);

}
}

