#pragma once

#include <memory>
#include <moto/core/fwd.hpp>

namespace moto::utils {

/// Rectangular complete-pivot LU in owned BLASFEO storage. No second
/// decomposition or numerical fallback; rank uses Eigen's default threshold.
class blasfeo_fullpiv_lu {
    struct impl;
    std::unique_ptr<impl> data_;

  public:
    blasfeo_fullpiv_lu();
    ~blasfeo_fullpiv_lu();
    blasfeo_fullpiv_lu(blasfeo_fullpiv_lu &&) noexcept;
    blasfeo_fullpiv_lu &operator=(blasfeo_fullpiv_lu &&) noexcept;
    blasfeo_fullpiv_lu(const blasfeo_fullpiv_lu &) = delete;
    blasfeo_fullpiv_lu &operator=(const blasfeo_fullpiv_lu &) = delete;

    /// Allocates lazily, copies A once, and refreshes the reusable factors.
    void compute(const matrix &a);
    size_t rank() const;
    matrix kernel();
    /// Reuses output storage across repeated stage linearizations.
    void kernel(matrix &output);
    /// Diagnostic reconstruction in original row/column order.
    matrix reconstruction() const;
    /// Output must have the required shape and must not alias rhs.
    void solve(Eigen::Ref<const matrix> rhs, Eigen::Ref<matrix> output);
    void transpose_solve(Eigen::Ref<const matrix> rhs, Eigen::Ref<matrix> output);
};

} // namespace moto::utils
