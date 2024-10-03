#ifndef __OPT_SPARSITY__
#define __OPT_SPARSITY__

#include <atri/core/fwd.hpp>

namespace atri {
enum class sparsity : int {
    dense = 0,
    diag,
    eye,
    scalar_eye,  // scalar * eye
};
class sparse_mat {
   private:
    sparsity type_;

   protected:
    matrix m_;

   public:
    size_t x0_, y0_, nrow_, ncol_;
    sparse_mat(size_t x0, size_t y0, size_t nrow, size_t ncol,
               sparsity type = sparsity::dense)
        : x0_(x0), y0_(y0), nrow_(nrow), ncol_(ncol), type_(type) {
        switch (type_) {
            case sparsity::dense:
                m_.resize(nrow_, ncol_);
                break;
            case sparsity::diag:
                assert(nrow_ == ncol_);
                m_.resize(nrow, 1);
                break;
            case sparsity::eye:
                assert(nrow_ == ncol_);
                break;
            case sparsity::scalar_eye:
                m_.resize(1, 1);
                assert(nrow_ == ncol_);
                break;
            default:
                break;
        }
    }
    // A.T @ D @ A
    virtual void diag_inner_product(Eigen::Ref<vector> dense_diag, Eigen::Ref<matrix> out) {
        // this way vectorization should still work(?)
        out.noalias() += m_.transpose() * dense_diag.asDiagonal() * m_;
    };
    // A.T @ rhs
    virtual void transpose_mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) {
        out.noalias() += m_.transpose() * dense_rhs;
    };
    // A @ rhs
    virtual void mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) {
        out.noalias() += m_ * dense_rhs;
    };
    // lhs @ A
    virtual void mult_by(Eigen::Ref<matrix> dense_lhs, Eigen::Ref<matrix> out) {
        out.noalias() += dense_lhs * m_;
    }
};
struct diag_mat final : public sparse_mat {
    diag_mat(size_t x0, size_t y0, size_t nrow, size_t ncol)
        : sparse_mat(x0, y0, nrow, ncol, sparsity::diag) {}
    void diag_inner_product(Eigen::Ref<vector> dense_diag, Eigen::Ref<matrix> out) override {
        out.diagonal().noalias() += m_.cwiseProduct(dense_diag.cwiseProduct(m_));
    };
    // A.T @ rhs
    void transpose_mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out.noalias() += m_.asDiagonal() * dense_rhs;
    };
    // A @ rhs
    void mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out.noalias() += m_.asDiagonal() * dense_rhs;
    };
    // lhs @ A
    void mult_by(Eigen::Ref<matrix> dense_lhs, Eigen::Ref<matrix> out) override {
        out.noalias() += dense_lhs * m_.asDiagonal();
    }
};
struct eye_mat final : public sparse_mat {
    eye_mat(size_t x0, size_t y0, size_t nrow, size_t ncol)
        : sparse_mat(x0, y0, nrow, ncol, sparsity::eye) {}
    void diag_inner_product(Eigen::Ref<vector> dense_diag, Eigen::Ref<matrix> out) override {
        out.diagonal() += dense_diag;
    };
    // A.T @ rhs
    void transpose_mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out += dense_rhs;
    };
    // A @ rhs
    void mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out += dense_rhs;
    };
    // lhs @ A
    void mult_by(Eigen::Ref<matrix> dense_lhs, Eigen::Ref<matrix> out) override {
        out += dense_lhs;
    }
};

struct scalar_eye_mat final : public sparse_mat {
    scalar_eye_mat(size_t x0, size_t y0, size_t nrow, size_t ncol)
        : sparse_mat(x0, y0, nrow, ncol, sparsity::scalar_eye) {}
    void diag_inner_product(Eigen::Ref<vector> dense_diag, Eigen::Ref<matrix> out) override {
        out.diagonal() += m_(0) * dense_diag;
    };
    // A.T @ rhs
    void transpose_mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out += m_(0) * dense_rhs;
    };
    // A @ rhs
    void mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out += m_(0) * dense_rhs;
    };
    // lhs @ A
    void mult_by(Eigen::Ref<matrix> dense_lhs, Eigen::Ref<matrix> out) override {
        out += m_(0) * dense_lhs;
    }
};
}  // namespace atri

#endif /*__SPARSITY_*/