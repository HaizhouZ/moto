#ifndef __FRLQR_FWD__
#define __FRLQR_FWD__

#include <Eigen/Core>
#include <memory>
#include <fmt/core.h>
#include <magic_enum.hpp>

namespace atri {
typedef float scalar_t;
typedef Eigen::Vector<scalar_t, -1> vector;
typedef const Eigen::Vector<scalar_t, -1> const_vector;
typedef Eigen::Matrix<scalar_t, -1, -1> matrix_cm; // column major
typedef Eigen::Matrix<scalar_t, -1, -1, Eigen::AutoAlign | Eigen::RowMajor> matrix_rm; // row major
typedef matrix_cm matrix;
using vector_ref = Eigen::Ref<Eigen::Matrix<scalar_t, -1, 1>>;

using mapped_vector = Eigen::Map<vector>;
using mapped_const_vector = Eigen::Map<const_vector>;
using mapped_matrix = Eigen::Map<matrix>;
}  // namespace atri

#endif /*__FWD_*/