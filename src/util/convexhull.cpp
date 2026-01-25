/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#include "convexhull.hh"
#include "mathutil.hh"

#include <Eigen/Dense>
#include <cmath>

namespace AGPLib
{
namespace Util
{

namespace
{

double nonneg_atan2(double x, double y)
{
    double atan_val = std::atan2(y, x);
    return atan_val >= 0 ? atan_val : std::max(0.0, 2.0 * PI + atan_val);
}

bool are_angles_equal(double alpha, double beta, double eps = EPS)
{
    if (mathutil::is_equal(alpha, beta, eps)) return true;
    // Check for 0 = 360° edge case
    if (mathutil::is_equal(alpha + 2.0 * PI, beta, eps) || mathutil::is_equal(alpha, beta + 2.0 * PI, eps)) return true;
    return false;
}

struct SweepedPoint
{
    size_t point_index;
    double angle;
    double sqdist_to_cog;

    SweepedPoint(size_t _index, double _angle, double _dist) : point_index(_index), angle(_angle), sqdist_to_cog(_dist) {};
};

using SweepSortResult = std::vector<SweepedPoint>;

SweepSortResult sweep_angle_sort(const std::vector<Vec2d>& points) {
    const size_t n = points.size();
    Vec2d cog = Vec2d::Zero();
    for (const Vec2d& p : points) 
    {
        cog += p;
    }
    cog /= static_cast<double>(n);
    std::vector<double> point_angles;
    point_angles.reserve(n);
    for (const Vec2d& p : points)
    {
        const Vec2d shifted_p = p - cog;
        point_angles.push_back(nonneg_atan2(shifted_p.x(), shifted_p.y()));
    }
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&point_angles](size_t i, size_t j) {
            return point_angles[i] < point_angles[j];
        });
    SweepSortResult res;
    res.reserve(n);
    for (size_t sort_index : indices)
    {
        res.emplace_back(sort_index, point_angles[sort_index], (points[sort_index] - cog).squaredNorm());
    }
    return res;
}

enum class CornerType
{
    Convex,
    Concave,
    Straight
};

CornerType determine_corner_type(const Vec2d& point_A, const Vec2d& point_B, const Vec2d& point_C, double eps = EPS)
{
    const Vec2d ab = point_B - point_A;
    const Vec2d bc = point_C - point_B;
    const double det = ab.x() * bc.y() - ab.y() * bc.x();
    if (mathutil::is_near_zero(det)) return CornerType::Straight;
    return det > 0 ? CornerType::Convex : CornerType::Concave;
}

}

std::vector<size_t> compute_convex_hull(const std::vector<Vec2d>& points, double eps)
{
    const size_t n = points.size();
    if (n == 0) return {};
    const SweepSortResult sweep_sort_info = sweep_angle_sort(points);

    // Find rightmost point
    size_t rightmost_sweep_index = 0;
    double max_x = -INF_DOUBLE;
    for (size_t i = 0; i < sweep_sort_info.size(); ++i)
    {
        const double x = points[sweep_sort_info[i].point_index].x();
        if (x > max_x)
        {
            max_x = x;
            rightmost_sweep_index = i;
        }
    }

    const double start_sweep_angle = sweep_sort_info[rightmost_sweep_index].angle;
    std::vector<size_t> convex_hull_sweep_indices = {rightmost_sweep_index};

    for (size_t i = 1; i <= n; i++)
    {
        const size_t current_sweep_index = (rightmost_sweep_index + i) % n;
        const double current_sweep_angle = sweep_sort_info[current_sweep_index].angle;
        // If sweep angle is equal to previous point, choose the one furthest aways from cog and continue
        const size_t prev_sweep_index = convex_hull_sweep_indices[convex_hull_sweep_indices.size() - 1];
        if (are_angles_equal(current_sweep_angle, sweep_sort_info[prev_sweep_index].angle, eps))
        {
            if (mathutil::is_clearly_greater(sweep_sort_info[current_sweep_index].sqdist_to_cog, sweep_sort_info[prev_sweep_index].sqdist_to_cog, eps))
            {
                convex_hull_sweep_indices[convex_hull_sweep_indices.size() - 1] = current_sweep_index;
            }
            continue;
        }
        // Complete convex hull up to current sweep index
        const Vec2d& current_point = points[sweep_sort_info[current_sweep_index].point_index];
        while (convex_hull_sweep_indices.size() > 1)
        {
            const size_t curr_hull_size = convex_hull_sweep_indices.size();
            const Vec2d& prev_point = points[sweep_sort_info[convex_hull_sweep_indices[curr_hull_size - 1]].point_index];
            const Vec2d& prev_prev_point = points[sweep_sort_info[convex_hull_sweep_indices[curr_hull_size - 2]].point_index];
            CornerType current_corner_type = determine_corner_type(prev_prev_point, prev_point, current_point, eps);
            if (current_corner_type == CornerType::Concave) 
                convex_hull_sweep_indices.pop_back();
            else break;
        }
        convex_hull_sweep_indices.push_back(current_sweep_index);
    }

    std::vector<size_t> convex_hull_indices;
    convex_hull_indices.reserve(convex_hull_sweep_indices.size());
    for (size_t sweep_index : convex_hull_sweep_indices)
    {
        convex_hull_indices.push_back(sweep_sort_info[sweep_index].point_index);
    }
    return convex_hull_indices;
}

}
}