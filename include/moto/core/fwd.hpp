#ifndef __FRLQR_FWD__
#define __FRLQR_FWD__

#include <Eigen/Core>
#include <cstddef> // For size_t
#include <fmt/core.h>
#include <memory>
#include <ranges>
#include <new>

#include <moto/utils/eigen_fmt.h>

namespace moto {
typedef double scalar_t;
typedef Eigen::Vector<scalar_t, -1> vector;
typedef Eigen::RowVector<scalar_t, -1> row_vector;
typedef const Eigen::Vector<scalar_t, -1> const_vector;
typedef Eigen::Matrix<scalar_t, -1, -1> matrix_cm;                                     // column major
typedef Eigen::Matrix<scalar_t, -1, -1, Eigen::AutoAlign | Eigen::RowMajor> matrix_rm; // row major
typedef matrix_cm matrix;
using vector_ref = Eigen::Ref<vector>;
using row_vector_ref = Eigen::Ref<row_vector>;
using matrix_ref = Eigen::Ref<matrix>;
using vector_const_ref = Eigen::Ref<const_vector>;

using mapped_vector = Eigen::Map<vector>;
using mapped_const_vector = Eigen::Map<const_vector>;
using mapped_matrix = Eigen::Map<matrix>;

#define def_ptr(name) typedef std::shared_ptr<name> name##_ptr_t
#define def_raw_ptr(name) typedef name* name##_ptr_t
#define def_ptr_named(name, type_name) typedef std::shared_ptr<type_name> name##_ptr_t
#define def_ptr_in_namespace(name, name_sp) typedef std::shared_ptr<name_sp::name> name##_ptr_t
#define def_unique_ptr(name) typedef std::unique_ptr<name> name##_ptr_t
#define def_unique_ptr_named(name, type_name) typedef std::unique_ptr<type_name> name##_ptr_t
inline auto to_matrix_ref_list(std::vector<matrix> &matrices) {
    std::vector<matrix_ref> refs;
    refs.reserve(matrices.size());
    for (auto &m : matrices) {
        refs.emplace_back(m);
    }
    return refs;
} // to_matrix_ref_list

inline constexpr auto range(size_t st, size_t ed) {
    return std::views::iota(st, ed);
}
inline constexpr auto range(size_t n) {
    return std::views::iota(size_t(0), n);
}
inline constexpr auto range_n(size_t st, size_t n) {
    return std::views::iota(st, st + n);
}

#define MOTO_ALIGN_NO_SHARING alignas(std::hardware_destructive_interference_size)
} // namespace moto

#endif /*__FWD_*/