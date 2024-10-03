#ifndef __OPT_SPARSITY__
#define __OPT_SPARSITY__

#include <atri/common/fwd.hpp>

namespace atri {
enum class sparsity : int {
    dense = 0,
    diag,
    eye,
};
struct sparsity_info {
    sparsity type_;
    size_t x0, y0, nrow, ncol;
};
struct sparse_mat {
    sparsity_info sp_;
    matrix m_;
    sparse_mat(const sparsity_info& sp)
        : sp_(sp) {
        switch (sp_.type_) {
            case sparsity::dense:
                m_.resize(sp_.nrow, sp_.ncol);
                break;
            case sparsity::diag:
                assert(sp_.nrow == sp_.ncol);
                m_.resize(sp_.nrow, 1);
                break;
            case sparsity::eye:
                assert(sp_.nrow == sp_.ncol);
                break;
            default:
                break;
        }
    }
    // A.T @ D @ A
    virtual void diag_inner_product(Eigen::Ref<vector> dense_diag, Eigen::Ref<matrix> out) {
        // this way vectorization should still work(?)
        out.noalias() = m_.transpose() * dense_diag.asDiagonal() * m_;
    };
    // A.T @ rhs
    virtual void transpose_mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) {
        out.noalias() = m_.transpose() * dense_rhs;
    };
    // A @ rhs
    virtual void mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) {
        out.noalias() = m_ * dense_rhs;
    };
    // lhs @ A
    virtual void mult_by(Eigen::Ref<matrix> dense_lhs, Eigen::Ref<matrix> out) {
        out.noalias() = dense_lhs * m_;
    }
};
struct diag_mat final : public sparse_mat {
    void diag_inner_product(Eigen::Ref<vector> dense_diag, Eigen::Ref<matrix> out) override {
        // this way vectorization should still work
        out.diagonal().noalias() = m_.cwiseProduct(dense_diag.cwiseProduct(m_));
    };
    // A.T @ rhs
    void transpose_mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out.noalias() = m_.asDiagonal() * dense_rhs;
    };
    // A @ rhs
    void mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out.noalias() = m_.asDiagonal() * dense_rhs;
    };
    // lhs @ A
    void mult_by(Eigen::Ref<matrix> dense_lhs, Eigen::Ref<matrix> out) override {
        out.noalias() = dense_lhs * m_.asDiagonal();
    }
};
struct eye_mat final : public sparse_mat {
    void diag_inner_product(Eigen::Ref<vector> dense_diag, Eigen::Ref<matrix> out) override {
        // this way vectorization should still work
        out.diagonal() = dense_diag;
    };
    // A.T @ rhs
    void transpose_mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out = dense_rhs;
    };
    // A @ rhs
    void mult(Eigen::Ref<matrix> dense_rhs, Eigen::Ref<matrix> out) override {
        out = dense_rhs;
    };
    // lhs @ A
    void mult_by(Eigen::Ref<matrix> dense_lhs, Eigen::Ref<matrix> out) override {
        out = dense_lhs;
    }
};
}  // namespace atri

#endif /*__SPARSITY_*/