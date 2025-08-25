#ifndef MOTO_UTILS_BLASFEO_FACTORIZATION_HPP
#define MOTO_UTILS_BLASFEO_FACTORIZATION_HPP

#include <blasfeo.h>
#include <moto/core/fwd.hpp>

namespace moto {
namespace utils {
struct buffer {
    matrix::AlignedMapType data_;
    void *mem_;
    size_t size_;

    buffer() : data_(nullptr, 0, 0), mem_(nullptr) {}
    void resize(size_t r, size_t c) {
        size_t required_size = r * c;
        if (size_ < required_size) {
            if (mem_ != nullptr)
                ::operator delete(mem_, std::align_val_t(64));
            mem_ = ::operator new(sizeof(scalar_t) * required_size, std::align_val_t(64));
            size_ = required_size;
        }
        if (r != data_.rows() || c != data_.cols())
            new (&data_) matrix::AlignedMapType(reinterpret_cast<scalar_t *>(mem_), r, c);
    }
    ~buffer() {
        if (mem_ != nullptr)
            ::operator delete(mem_, std::align_val_t(64));
    }
};
struct blasfeo_buffer {
    blasfeo_dmat data_;
    void *mem_ = nullptr;
    size_t size_ = 0;

    blasfeo_buffer() : mem_(nullptr) {}
    void resize(size_t r, size_t c) {
        size_t required_size = blasfeo_memsize_dmat(r, c);
        if (size_ < required_size) {
            if (mem_ != nullptr)
                v_free_align(mem_);
            int B_size = blasfeo_memsize_dmat(r, c); // size of memory needed by B
            v_zeros_align(&mem_, B_size);
            size_ = required_size;
        }
        blasfeo_create_dmat(r, c, &data_, mem_);
    }
    ~blasfeo_buffer() {
        if (mem_ != nullptr)
            v_free_align(mem_);
    }
    // Helper function to convert Eigen::MatrixXd to blasfeo_dmat.
    // This allocates and packs the data into a BLASFEO matrix structure.
    template <typename T>
    void from_eigen(const T &eigen_mat) {
        // Pack the data from the Eigen matrix into the BLASFEO matrix
        resize(eigen_mat.rows(), eigen_mat.cols());
        blasfeo_pack_dmat(eigen_mat.rows(), eigen_mat.cols(), (double *)eigen_mat.data(), eigen_mat.rows(), &data_, 0, 0);
    }
    // Helper function to convert blasfeo_dmat to Eigen::MatrixXd.
    // This unpacks data from a BLASFEO matrix into an Eigen matrix.
    template <typename T>
    void to_eigen(T &eigen_mat) {
        // Resize the Eigen matrix to match BLASFEO matrix dimensions
        eigen_mat.resize(data_.m, data_.n);
        // Unpack the data from the BLASFEO matrix into the Eigen matrix
        blasfeo_unpack_dmat(data_.m, data_.n, &data_, 0, 0, (double *)eigen_mat.data(), eigen_mat.rows());
    }
};

struct blasfeo_llt {
    blasfeo_buffer L_, U_; // Lower triangular matrix from LLT
    blasfeo_buffer rhs_, res_;

    bool valid() const {
        for(size_t i = 0; i < L_.data_.m; i++) {
            for (size_t j = 0; j <= i; j++) {
                double& v = BLASFEO_DMATEL(&L_.data_, i, j);
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
    }
    template <typename rhs_type, typename res_type>
    void solve(const rhs_type &b, res_type &a, double alpha = 1.0) {
        size_t size = L_.data_.m;
        assert(b.rows() == size && a.rows() == size);
        assert(b.cols() == a.cols());
        rhs_.from_eigen(b);
        res_.resize(size, b.cols());
        blasfeo_dtrsm_llnn(size, b.cols(), alpha, &L_.data_, 0, 0, &rhs_.data_, 0, 0, &res_.data_, 0, 0);
        blasfeo_dtrtr_l(size, &L_.data_, 0, 0, &U_.data_, 0, 0);
        blasfeo_dtrsm_lunn(size, b.cols(), 1, &U_.data_, 0, 0, &res_.data_, 0, 0, &res_.data_, 0, 0);
        res_.to_eigen(a);
    }
};

} // namespace utils
} // namespace moto

#endif