#pragma once

#include <Eigen/LU>
#include <moto/utils/blasfeo_factorizer/blasfeo_fullpiv_lu.hpp>
#include <stdexcept>

namespace moto::solver::ns_riccati {

enum class equality_projection_backend { eigen, panel_lu };

/// Captures the backend at factor refresh. Corrections and dual recovery reuse
/// the same factors even if an outer setting has subsequently changed.
class hard_equality_factor {
    Eigen::FullPivLU<matrix> eigen_;
    utils::blasfeo_fullpiv_lu panel_;
    equality_projection_backend backend_ = equality_projection_backend::eigen;

  public:
    void compute(const matrix &a, equality_projection_backend backend) {
        switch (backend) {
        case equality_projection_backend::eigen:
            eigen_.compute(a);
            break;
        case equality_projection_backend::panel_lu:
            panel_.compute(a);
            break;
        default:
            throw std::invalid_argument("Unknown equality projection backend");
        }
        backend_ = backend;
    }
    size_t rank() const {
        return backend_ == equality_projection_backend::eigen ? eigen_.rank() : panel_.rank();
    }
    void kernel(matrix &output) {
        if (backend_ == equality_projection_backend::eigen)
            output = eigen_.kernel();
        else
            panel_.kernel(output);
    }
    template <typename Rhs, typename Output> void solve(const Rhs &rhs, Output &output) {
        if (backend_ == equality_projection_backend::eigen)
            output.noalias() = eigen_.solve(rhs);
        else
            panel_.solve(rhs, output);
    }
    template <typename Rhs, typename Output> void transpose_solve(const Rhs &rhs, Output &output) {
        if (backend_ == equality_projection_backend::eigen)
            output.noalias() = eigen_.transpose().solve(rhs);
        else
            panel_.transpose_solve(rhs, output);
    }
};

} // namespace moto::solver::ns_riccati
