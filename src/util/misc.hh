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

Vec3d any_orthogonal(const Vec3d& _p);

Vec3d make_orthogonal_dir(const Vec3d& v, const Vec3d& normal);

template <class T>
inline void hash_combine(std::size_t& hash, const T& v)
{
	// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
	std::hash<T> hasher;
	hash ^= hasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

}
}

