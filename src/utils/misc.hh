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

Vec3d any_orthogonal(const Vec3d& _p);

Vec3d make_orthogonal_dir(const Vec3d& v, const Vec3d& normal);

}

