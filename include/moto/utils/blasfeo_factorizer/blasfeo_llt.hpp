#ifndef MOTO_UTILS_BLASFEO_LLT_HPP
#define MOTO_UTILS_BLASFEO_LLT_HPP

#include <blasfeo.h>
#include <moto/utils/blasfeo_factorizer/blasfeo_buffer.hpp>

namespace moto {
namespace utils {
struct blasfeo_llt {
    blasfeo_buffer L_, U_; // Lower triangular matrix from LLT
    blasfeo_buffer I_;     // Identity matrix for computing inverse
    blasfeo_buffer rhs_, res_;
    blasfeo_vec_buffer vec_rhs_, vec_res_;

    bool valid() const {
        for (size_t i = 0; i < L_.data_.m; i++) {
            for (size_t j = 0; j <= i; j++) {
                double &v = BLASFEO_DMATEL(&L_.data_, i, j);
                if (std::isnan(v) || std::isinf(v))
                    return false;
            }
        }
        return true;
    }
    void compute(matrix &A) {
        L_.from_eigen(A);
        U_.resize(L_.data_.m, L_.data_.n);
        blasfeo_dpotrf_l(A.rows(), &L_.data_, 0, 0, &L_.data_, 0, 0);
        blasfeo_dtrtr_l(A.rows(), &L_.data_, 0, 0, &U_.data_, 0, 0);
    }
    template <typename rhs_type, typename res_type>
    void solve(const rhs_type &b, res_type &a, double alpha = 1.0) {
        size_t size = L_.data_.m;
        assert(b.rows() == size && a.rows() == size);
        assert(b.cols() == a.cols());
        if constexpr (rhs_type::ColsAtCompileTime == 1) {
            vec_rhs_.from_eigen(b);
            vec_res_.resize(size, 1);
            blasfeo_dtrsv_lnn(size, &L_.data_, 0, 0, &vec_rhs_.data_, 0, &vec_res_.data_, 0);
            blasfeo_dtrsv_unn(size, &U_.data_, 0, 0, &vec_res_.data_, 0, &vec_res_.data_, 0);
            blasfeo_dvecsc(size, alpha, &vec_res_.data_, 0);
            vec_res_.to_eigen(a);
        } else {
            rhs_.from_eigen(b);
            res_.resize(size, b.cols());
            blasfeo_dtrsm_llnn(size, b.cols(), alpha, &L_.data_, 0, 0, &rhs_.data_, 0, 0, &res_.data_, 0, 0);
            blasfeo_dtrsm_lunn(size, b.cols(), 1, &U_.data_, 0, 0, &res_.data_, 0, 0, &res_.data_, 0, 0);
            res_.to_eigen(a);
        }
    }
    template <typename res_type>
    void get_inverse(res_type &a, double alpha = 1.0) {
        assert(a.rows() == L_.data_.m && a.cols() == L_.data_.n);
        size_t size = L_.data_.m;
        I_.resize(size, size);
        for (size_t i = 0; i < size; i++)
            BLASFEO_DMATEL(&I_.data_, i, i) = 1.0;
        res_.resize(size, size);
        blasfeo_dtrsm_llnn(size, size, alpha, &L_.data_, 0, 0, &I_.data_, 0, 0, &res_.data_, 0, 0);
        blasfeo_dtrsm_lunn(size, size, 1, &U_.data_, 0, 0, &res_.data_, 0, 0, &res_.data_, 0, 0);
        res_.to_eigen(a);
    }
};

} // namespace utils
} // namespace moto

#endif