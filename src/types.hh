/*
 * This file is part of the agplib library
 * (https://github.com/AntonFlorey/agplib)
 * and is released under the MIT license.
 *
 * Authors: Anton Florey
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace AGPLib
{

// Scalar constants
constexpr double NAN_DOUBLE = std::numeric_limits<double>::quiet_NaN();
constexpr double INF_DOUBLE = std::numeric_limits<double>::infinity();
constexpr double PI = 3.14159265358979323846;
constexpr double EPS = 1e-12;

// Fixed-size vectors
template <int d, typename T> using Vec = Eigen::Matrix<T, d, 1>;
template <typename T> using Vec2 = Vec<2, T>;
template <typename T> using Vec3 = Vec<3, T>;
using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;

// Fixed-size matrices
template <int rows, int cols, typename T> using Mat = Eigen::Matrix<T, rows, cols>;
template <typename T> using Mat2 = Mat<2, 2, T>;
template <typename T> using Mat3 = Mat<3, 3, T>;
using Mat2d = Mat2<double>;
using Mat3d = Mat3<double>;

// Dynamic-size matrices
using MatXd = Eigen::MatrixXd;

// Dynamic-size vectors
template <typename T> using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using VecXd = Eigen::VectorXd;

// Sparse matrices
using Triplet = Eigen::Triplet<double>;
using SparseMatrix = Eigen::SparseMatrix<double>;

// Diagonal matrices
using DiagonalMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

}