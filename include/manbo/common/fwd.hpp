#ifndef __FRLQR_FWD__
#define __FRLQR_FWD__

#include <Eigen/Core>
#include <fmt/core.h>
#include <magic_enum.hpp>

namespace manbo {
typedef float scalar_t;
typedef Eigen::Vector<scalar_t, -1> vector;
typedef Eigen::Matrix<scalar_t, -1, -1> matrix;
}  // namespace manbo

#endif /*__FWD_*/