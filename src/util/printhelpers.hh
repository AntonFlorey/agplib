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

template <typename T>
std::string id_container_to_string(const T& id_container)
{
    std::string res = "{";
    for (const auto& id : id_container)
    {
        res += std::to_string(id) + ", ";
    }
    // remove last comma and space
    if (!id_container.empty())
    {
        res.pop_back();
        res.pop_back();
    }
    res += "}";
    return res;
}

}
}