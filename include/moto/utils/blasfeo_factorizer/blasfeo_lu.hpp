#ifndef MOTO_UTILS_BLASFEO_LU_HPP
#define MOTO_UTILS_BLASFEO_LU_HPP

#include <moto/utils/blasfeo_factorizer/blasfeo_buffer.hpp>

namespace moto {
namespace utils {
struct blasfeo_lu {
    blasfeo_buffer D_, LU_, LUt_; // Lower triangular matrix from LU
    bool tr_updated_ = false;
    int *ipiv = nullptr;
    size_t ipiv_size = 0;
    blasfeo_lu() : ipiv(nullptr), ipiv_size(0) {}
    ~blasfeo_lu() {
        if (ipiv != nullptr)
            free(ipiv);
    }

    bool valid() const {
        for (size_t i = 0; i < LU_.data_.m; i++) {
            for (size_t j = 0; j <= LU_.data_.n; j++) {
                double &v = BLASFEO_DMATEL(&LU_.data_, i, j);
                if (std::isnan(v) || std::isinf(v))
                    return false;
            }
        }
        return true;
    }
    template <typename T>
    void compute(T &A) {
        assert(A.rows() == A.cols());
        D_.from_eigen(A);
        LU_.resize(D_.data_.m, D_.data_.n);
        if (ipiv_size < (size_t)A.rows()) {
            if (ipiv != nullptr)
                free(ipiv);
            ipiv_size = A.rows();
            ipiv = (int *)malloc(sizeof(int) * ipiv_size);
        }
        blasfeo_dgetrf_rp(A.rows(), A.cols(), &D_.data_, 0, 0, &LU_.data_, 0, 0, ipiv);
        tr_updated_ = false;
    }
    template <typename rhs_type, typename res_type>
    void solve(const rhs_type &rhs, res_type &res, double alpha = 1.0) {
        size_t size = D_.data_.m;
        assert(rhs.rows() == size && res.rows() == size);
        assert(rhs.cols() == res.cols());
        if constexpr (rhs_type::ColsAtCompileTime == 1) {
            thread_local blasfeo_vec_buffer vec_rhs_, vec_res_;
            vec_rhs_.from_eigen(rhs);
            vec_res_.resize(size, 1);
            blasfeo_dvecpe(size, ipiv, &vec_rhs_.data_, 0);
            blasfeo_dtrsv_lnu(size, &LU_.data_, 0, 0, &vec_rhs_.data_, 0, &vec_res_.data_, 0);
            blasfeo_dtrsv_unn(size, &LU_.data_, 0, 0, &vec_res_.data_, 0, &vec_res_.data_, 0);
            blasfeo_dvecsc(size, alpha, &vec_res_.data_, 0);
            vec_res_.to_eigen(res);
        } else {
            thread_local blasfeo_buffer rhs_, res_;
            rhs_.from_eigen(rhs);
            res_.resize(size, rhs.cols());
            blasfeo_drowpe(size, ipiv, &rhs_.data_);
            blasfeo_dtrsm_llnu(size, rhs.cols(), alpha, &LU_.data_, 0, 0, &rhs_.data_, 0, 0, &res_.data_, 0, 0);
            blasfeo_dtrsm_lunn(size, rhs.cols(), 1, &LU_.data_, 0, 0, &res_.data_, 0, 0, &res_.data_, 0, 0);
            res_.to_eigen(res);
        }
    }
    template <typename rhs_type, typename res_type>
    void transpose_solve(const rhs_type &rhs, res_type &res, double alpha = 1.0) {
        size_t size = D_.data_.m;
        assert(rhs.rows() == size && res.rows() == size);
        assert(rhs.cols() == res.cols());
        if constexpr (rhs_type::ColsAtCompileTime == 1) {
            thread_local blasfeo_vec_buffer vec_rhs_, vec_res_;
            vec_rhs_.from_eigen(rhs);
            vec_res_.resize(size, 1);
            blasfeo_dtrsv_utn(size, &LU_.data_, 0, 0, &vec_rhs_.data_, 0, &vec_res_.data_, 0);
            blasfeo_dtrsv_ltu(size, &LU_.data_, 0, 0, &vec_res_.data_, 0, &vec_res_.data_, 0);
            blasfeo_dvecpei(size, ipiv, &vec_res_.data_, 0);
            blasfeo_dvecsc(size, alpha, &vec_res_.data_, 0);
            vec_res_.to_eigen(res);
        } else {
            thread_local blasfeo_buffer rhs_, res_;
            if (!tr_updated_) {
                LUt_.resize(size, size);
                blasfeo_dgetr(size, size, &LU_.data_, 0, 0, &LUt_.data_, 0, 0);
                tr_updated_ = true;
            }
            rhs_.from_eigen(rhs);
            res_.resize(size, rhs.cols());
            blasfeo_dtrsm_llnn(size, rhs.cols(), alpha, &LUt_.data_, 0, 0, &rhs_.data_, 0, 0, &res_.data_, 0, 0);
            blasfeo_dtrsm_lunu(size, rhs.cols(), 1, &LUt_.data_, 0, 0, &res_.data_, 0, 0, &res_.data_, 0, 0);
            blasfeo_drowpei(size, ipiv, &res_.data_);
            res_.to_eigen(res);
        }
    }
};
} // namespace utils
} // namespace moto
#endif