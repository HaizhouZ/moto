#ifndef MOTO_UTILS_BLASFEO_LQ_HPP
#define MOTO_UTILS_BLASFEO_LQ_HPP

#include <moto/utils/blasfeo_factorizer/blasfeo_buffer.hpp>

namespace moto {
namespace utils {
struct blasfeo_qr {
    blasfeo_buffer D_, LQ_, Q_, L_, LUt_; // Lower triangular matrix from LU
    blasfeo_buffer rhs_, res_;
    bool tr_updated_ = false;
    size_t n, m;
    void *lq_work_mem = nullptr;
    void *orglq_work_mem = nullptr;
    size_t lq_work_size = 0;
    size_t orglq_work_size = 0;
    size_t nzpiv_ = 0; // number of non-zero pivots

    ~blasfeo_qr() {
        if (lq_work_mem != nullptr)
            v_free_align(lq_work_mem);
        if (orglq_work_mem != nullptr)
            v_free_align(orglq_work_mem);
    }

    void resize(size_t rows, size_t cols) {
        n = cols;
        m = rows;
        size_t lq_work_size_new = blasfeo_dgelqf_worksize(rows, cols);
        if (lq_work_size < lq_work_size_new) {
            if (lq_work_mem != nullptr)
                v_free_align(lq_work_mem);
            v_zeros_align(&lq_work_mem, lq_work_size_new);
            lq_work_size = lq_work_size_new;
        }
        assert(rows <= cols);
        size_t orglq_work_size_new = blasfeo_dorglq_worksize(cols, cols, rows);
        if (orglq_work_size < orglq_work_size_new) {
            if (orglq_work_mem != nullptr)
                v_free_align(orglq_work_mem);
            v_zeros_align(&orglq_work_mem, orglq_work_size_new);
            orglq_work_size = orglq_work_size_new;
        }
    }

    template <typename T>
    void Q(T &A) { Q_.to_eigen(A); }

    template <typename T>
    void L(T &A) { L_.to_eigen(A); }

    size_t nonzero_pivots() const { return nzpiv_; }

    bool valid() const {
        for (size_t i = 0; i < LQ_.data_.m; i++) {
            for (size_t j = 0; j <= LQ_.data_.n; j++) {
                double &v = BLASFEO_DMATEL(&LQ_.data_, i, j);
                if (std::isnan(v) || std::isinf(v))
                    return false;
            }
        }
        return true;
    }
    template <typename T>
    void compute(T &A) {
        D_.from_eigen(A);
        LQ_.resize(D_.data_.m, D_.data_.n);
        resize(A.rows(), A.cols());
        blasfeo_dgelqf(A.rows(), A.cols(), &D_.data_, 0, 0, &LQ_.data_, 0, 0, lq_work_mem);
        L_.resize(m, m);
        Q_.resize(n, n);
        blasfeo_dtrcp_l(m, &LQ_.data_, 0, 0, &L_.data_, 0, 0);
        blasfeo_dorglq(n, n, m, &LQ_.data_, 0, 0, &Q_.data_, 0, 0, orglq_work_mem);
        tr_updated_ = false;
    }
    template <typename rhs_type, typename res_type>
    void solve(const rhs_type &rhs, res_type &res, double alpha = 1.0) {
        size_t size = m;
        size_t cols = n;
        assert(rhs.rows() == size && res.rows() == cols);
        assert(rhs.cols() == res.cols());
        rhs_.from_eigen(rhs);

        res_.resize(cols, rhs.cols());
        blasfeo_dtrsm_llnn(m, rhs.cols(), alpha, &L_.data_, 0, 0, &rhs_.data_, 0, 0, &rhs_.data_, 0, 0);
        blasfeo_dgemm_tn(n, rhs.cols(), m, 1.0, &Q_.data_, 0, 0, &rhs_.data_, 0, 0, 0.0, &rhs_.data_, 0, 0, &res_.data_, 0, 0);

        res_.to_eigen(res);
    }

    void residual(matrix &res) {
        res_.resize(m, n);
        blasfeo_dgemm_nn(m, n, m, 1.0, &LQ_.data_, 0, 0, &Q_.data_, 0, 0, -1.0, &D_.data_, 0, 0, &res_.data_, 0, 0);
        res_.to_eigen(res);
    }
    // template <typename rhs_type, typename res_type>
    // void transpose_solve(const rhs_type &rhs, res_type &res, double alpha = 1.0) {
    //     size_t size = D_.data_.m;
    //     assert(rhs.rows() == size && res.rows() == size);
    //     assert(rhs.cols() == res.cols());
    //     rhs_.from_eigen(rhs);
    //     res_.resize(size, rhs.cols());
    //     if (!tr_updated_) {
    //         LUt_.resize(size, size);
    //         blasfeo_dgetr(size, size, &LQ_.data_, 0, 0, &LUt_.data_, 0, 0);
    //         tr_updated_ = true;
    //     }
    //     blasfeo_dtrsm_llnn(size, rhs.cols(), alpha, &LUt_.data_, 0, 0, &rhs_.data_, 0, 0, &res_.data_, 0, 0);
    //     blasfeo_dtrsm_lunu(size, rhs.cols(), 1, &LUt_.data_, 0, 0, &res_.data_, 0, 0, &res_.data_, 0, 0);
    //     blasfeo_drowpei(size, ipiv, &res_.data_);
    //     res_.to_eigen(res);
    // }
};
} // namespace utils
} // namespace moto
#endif