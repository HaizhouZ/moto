#ifndef MOTO_UTILS_BLASFEO_LQ_HPP
#define MOTO_UTILS_BLASFEO_LQ_HPP

#include <moto/utils/blasfeo_factorizer/blasfeo_buffer.hpp>

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace moto::utils {

/**
 * Full-row-rank LQ factorization used by the NSP hard-equality projection.
 *
 * The ordinary path factors A = [L 0] Q with BLASFEO.  FullPivLU is retained
 * only as a rank-revealing fallback because dgelqf is intentionally unpivoted.
 */
class blasfeo_lq {
  public:
    blasfeo_lq() = default;
    blasfeo_lq(const blasfeo_lq &) = delete;
    blasfeo_lq &operator=(const blasfeo_lq &) = delete;

    ~blasfeo_lq() {
        if (lq_work_)
            v_free_align(lq_work_);
        if (orglq_work_)
            v_free_align(orglq_work_);
    }

    template <typename matrix_type>
    void compute(const matrix_type &input) {
        rows_ = static_cast<size_t>(input.rows());
        cols_ = static_cast<size_t>(input.cols());
        fast_path_ = false;
        rank_ = 0;

        // dorglq requires k <= min(m, n).  Preserve the previous generic
        // behavior for over-constrained geometry through the fallback.
        if (rows_ > cols_) {
            fallback_.compute(input);
            rank_ = static_cast<size_t>(fallback_.rank());
            return;
        }

        resize_workspaces(rows_, cols_);
        input_.from_eigen(input);
        lq_.resize(rows_, cols_);
        blasfeo_dgelqf(static_cast<int>(rows_), static_cast<int>(cols_),
                       &input_.data_, 0, 0, &lq_.data_, 0, 0, lq_work_);

        double max_pivot = 0.;
        bool finite = true;
        for (size_t i = 0; i < rows_; ++i) {
            const double pivot = BLASFEO_DMATEL(&lq_.data_, i, i);
            finite = finite && std::isfinite(pivot);
            max_pivot = std::max(max_pivot, std::abs(pivot));
        }
        const double tolerance =
            std::numeric_limits<double>::epsilon() *
            static_cast<double>(std::max(rows_, cols_)) * max_pivot;
        for (size_t i = 0; i < rows_; ++i) {
            const double pivot = BLASFEO_DMATEL(&lq_.data_, i, i);
            finite = finite && std::abs(pivot) > tolerance;
        }

        if (!finite || max_pivot == 0.) {
            fallback_.compute(input);
            rank_ = static_cast<size_t>(fallback_.rank());
            return;
        }

        lower_.resize(rows_, rows_);
        blasfeo_dtrcp_l(static_cast<int>(rows_), &lq_.data_, 0, 0,
                        &lower_.data_, 0, 0);
        orthogonal_.resize(cols_, cols_);
        blasfeo_dorglq(static_cast<int>(cols_), static_cast<int>(cols_),
                       static_cast<int>(rows_), &lq_.data_, 0, 0,
                       &orthogonal_.data_, 0, 0, orglq_work_);
        rank_ = rows_;
        fast_path_ = true;
    }

    [[nodiscard]] size_t rank() const { return rank_; }
    [[nodiscard]] bool uses_blasfeo() const { return fast_path_; }

    template <typename result_type>
    void kernel(result_type &result) {
        if (!fast_path_) {
            result = fallback_.kernel();
            return;
        }
        const size_t nullity = cols_ - rows_;
        result.resize(static_cast<Eigen::Index>(cols_),
                      static_cast<Eigen::Index>(nullity));
        if (!nullity)
            return;
        kernel_.resize(cols_, nullity);
        // Q^T[:, m:n] is the transpose of Q[m:n, :].
        blasfeo_dgetr(static_cast<int>(nullity), static_cast<int>(cols_),
                      &orthogonal_.data_, static_cast<int>(rows_), 0,
                      &kernel_.data_, 0, 0);
        kernel_.to_eigen(result);
    }

    template <typename rhs_type, typename result_type>
    void solve(const rhs_type &rhs, result_type &result) {
        if (!fast_path_) {
            result = fallback_.solve(rhs);
            return;
        }
        if (static_cast<size_t>(rhs.rows()) != rows_)
            throw std::invalid_argument("LQ solve right-hand side row mismatch");
        rhs_.from_eigen(rhs);
        triangular_.resize(rows_, static_cast<size_t>(rhs.cols()));
        solution_.resize(cols_, static_cast<size_t>(rhs.cols()));
        blasfeo_dtrsm_llnn(static_cast<int>(rows_), rhs.cols(), 1.,
                           &lower_.data_, 0, 0, &rhs_.data_, 0, 0,
                           &triangular_.data_, 0, 0);
        blasfeo_dgemm_tn(static_cast<int>(cols_), rhs.cols(),
                         static_cast<int>(rows_), 1., &orthogonal_.data_, 0, 0,
                         &triangular_.data_, 0, 0, 0., &solution_.data_, 0, 0,
                         &solution_.data_, 0, 0);
        solution_.to_eigen(result);
    }

    template <typename rhs_type, typename result_type>
    void transpose_solve(const rhs_type &rhs, result_type &result) {
        if (!fast_path_) {
            result = fallback_.transpose().solve(rhs);
            return;
        }
        if (static_cast<size_t>(rhs.rows()) != cols_)
            throw std::invalid_argument(
                "LQ transpose solve right-hand side row mismatch");
        rhs_.from_eigen(rhs);
        triangular_.resize(rows_, static_cast<size_t>(rhs.cols()));
        solution_.resize(rows_, static_cast<size_t>(rhs.cols()));
        blasfeo_dgemm_nn(static_cast<int>(rows_), rhs.cols(),
                         static_cast<int>(cols_), 1., &orthogonal_.data_, 0, 0,
                         &rhs_.data_, 0, 0, 0., &triangular_.data_, 0, 0,
                         &triangular_.data_, 0, 0);
        blasfeo_dtrsm_lltn(static_cast<int>(rows_), rhs.cols(), 1.,
                           &lower_.data_, 0, 0, &triangular_.data_, 0, 0,
                           &solution_.data_, 0, 0);
        solution_.to_eigen(result);
    }

  private:
    void resize_workspaces(size_t rows, size_t cols) {
        const size_t lq_size = blasfeo_dgelqf_worksize(
            static_cast<int>(rows), static_cast<int>(cols));
        if (lq_size > lq_work_size_) {
            if (lq_work_)
                v_free_align(lq_work_);
            v_zeros_align(&lq_work_, lq_size);
            lq_work_size_ = lq_size;
        }
        const size_t orglq_size = blasfeo_dorglq_worksize(
            static_cast<int>(cols), static_cast<int>(cols),
            static_cast<int>(rows));
        if (orglq_size > orglq_work_size_) {
            if (orglq_work_)
                v_free_align(orglq_work_);
            v_zeros_align(&orglq_work_, orglq_size);
            orglq_work_size_ = orglq_size;
        }
    }

    blasfeo_buffer input_, lq_, lower_, orthogonal_;
    blasfeo_buffer rhs_, triangular_, solution_, kernel_;
    Eigen::FullPivLU<matrix> fallback_;
    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t rank_ = 0;
    bool fast_path_ = false;
    void *lq_work_ = nullptr;
    void *orglq_work_ = nullptr;
    size_t lq_work_size_ = 0;
    size_t orglq_work_size_ = 0;
};

} // namespace moto::utils

#endif
