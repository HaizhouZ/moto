#ifndef MOTO_SPMM_IMPL_HPP
#define MOTO_SPMM_IMPL_HPP

#include <moto/spmm/panel_mat.hpp>

namespace moto {

template <typename expr_type>
struct is_sp_expr {
    static constexpr bool value = false;
};

template <typename... Args>
struct is_diag_type {
    static constexpr bool value = false;
};

template <typename expr_type>
struct is_diag_type<const Eigen::DiagonalWrapper<expr_type>> {
    static constexpr bool value = true;
};

template <typename expr_type>
struct is_diag_type<Eigen::DiagonalWrapper<expr_type>> {
    static constexpr bool value = true;
};

struct eval_config {
    bool sym_res = false;
    bool same_sides = false;
    bool dst_panel = false;
    bool overwrite = false;
    bool add_to = true;
};

template <typename expr_type>
    requires(is_sp_expr<expr_type>::value == false and !std::is_base_of_v<panel_mat_base, expr_type>)
struct sp_expr : public expr_type, public panel_mat<is_diag_type<expr_type>::value ? sparsity::diag : sparsity::dense> {
    using base = panel_mat<is_diag_type<expr_type>::value ? sparsity::diag : sparsity::dense>;
    sp_expr(expr_type &&expr, std::pair<size_t, size_t> st)
        : expr_type(std::move(expr)),
          base(st.first, st.second, expr_type::rows(), expr_type::cols(), false) {
    }
    using base::col_ed_;
    using base::col_st_;
    using base::cols_;
    using base::row_ed_;
    using base::row_st_;
    using base::rows_;

    template <eval_config config = eval_config(), typename mat_type, typename cache_type>
        requires(!std::is_base_of_v<panel_mat_base, mat_type> && !std::is_base_of_v<panel_mat_base, cache_type>)
    void eval_then_add_to(mat_type &dst, cache_type &cache) const {
        size_t out_row_st = row_st_;
        size_t out_col_st = col_st_;
        if constexpr (config.dst_panel) {
            out_row_st = out_col_st = 0; // panel_mat_base does not have row_st_ and col_st_
            assert(dst.rows() == rows_ && dst.cols() == cols_ && "output matrix size mismatch");
        } else
            assert(dst.rows() >= row_ed_ && dst.cols() >= col_ed_ && "output matrix size mismatch");
        if constexpr (is_diag_type<expr_type>::value) {
            throw std::runtime_error("Not implemented");
            auto out_block = dst.block(out_row_st, out_col_st, this->rows_, this->cols_);
            out_block.diagonal() += static_cast<const expr_type &>(*this).diagonal();
            if constexpr (config.sym_res && !config.same_sides) {
                auto sym_block = dst.block(out_col_st, out_row_st, this->cols_, this->rows_);
                sym_block.diagonal() = out_block.diagonal();
            }
        } else {
            auto add_to_ = [&, this](auto &&out_block, auto &&sym_block) {
                if constexpr (config.sym_res) {
                    bool sym_alias = (this->col_st_ <= this->row_st_ && this->col_ed_ >= this->row_st_) ||
                                     (this->col_st_ <= this->row_ed_ && this->col_ed_ >= this->row_ed_) ||
                                     (this->row_st_ <= this->col_st_ && this->row_ed_ >= this->col_st_) ||
                                     (this->row_st_ <= this->col_ed_ && this->row_ed_ >= this->col_ed_);
                    static_assert(config.add_to, "Symmetric result requires add_to to be true");
                    if (!sym_alias) {
                        // std::cout << "No aliasing detected, copying. col_st_ = " << col_st_
                        //           << ", row_st_ = " << row_st_
                        //           << ", col_ed_ = " << col_ed_
                        //           << ", row_ed_ = " << row_ed_ << std::endl;
                        out_block.noalias() += static_cast<const expr_type &>(*this);
                        sym_block.noalias() = out_block.transpose();
                    } else {                               // aliasing
                        if constexpr (config.same_sides) { // no need to copy
                                                           //     std::cout << "Aliasing detected, but same sides no need to copy\n";
                            out_block.noalias() += static_cast<const expr_type &>(*this);
                        } else { // need to copy
                            // std::cout << "Aliasing detected, need to copy cache\n";
                            cache.noalias() = static_cast<const expr_type &>(*this);
                            out_block.noalias() += cache;
                            sym_block.noalias() += cache.transpose();
                        }
                    }
                } else {
                    // std::cout << "No symmetry, adding to output block\n";
                    if constexpr (config.overwrite) {
                        if constexpr (config.add_to)
                            out_block.noalias() = static_cast<const expr_type &>(*this);
                        else
                            out_block.noalias() = -static_cast<const expr_type &>(*this);
                    } else {
                        if constexpr (config.add_to)
                            out_block.noalias() += static_cast<const expr_type &>(*this);
                        else
                            out_block.noalias() -= static_cast<const expr_type &>(*this);
                    }
                }
            };
            if (this->cols_ == dst.cols() && this->rows_ == dst.rows())
                add_to_(dst, dst);
            else if (this->cols_ == dst.cols()) {
                add_to_(dst.middleRows(out_row_st, this->rows_), dst.middleCols(out_row_st, this->rows_));
            } else if (this->rows_ == dst.rows()) {
                add_to_(dst.middleCols(out_col_st, this->cols_), dst.middleRows(out_col_st, this->cols_));
            } else {
                add_to_(dst.block(out_row_st, out_col_st, this->rows_, this->cols_),
                        dst.block(out_col_st, out_row_st, this->cols_, this->rows_));
            }
        }
    }
};

template <typename expr_type>
struct is_diag_type<sp_expr<expr_type>> {
    static constexpr bool value = is_diag_type<expr_type>::value;
};

template <sparsity Sp>
struct is_diag_type<panel_mat<Sp>> {
    static constexpr bool value = Sp == sparsity::diag || Sp == sparsity::eye;
};

template <typename expr_type>
struct is_sp_expr<sp_expr<expr_type>> {
    static constexpr bool value = true;
};

struct non_op {
    int dim = 0;
};

template <>
struct sp_expr<non_op> : public panel_mat<sparsity::eye> {
    using base = panel_mat<sparsity::eye>;
    sp_expr(non_op &&n, std::pair<size_t, size_t> st) : base(st.first, st.second, n.dim, n.dim, false) {}
    using base::col_ed_;
    using base::col_st_;
    using base::cols_;
    using base::row_ed_;
    using base::row_st_;
    using base::rows_;
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    template <eval_config config = eval_config(), typename mat_type, typename cache_type>
        requires(!std::is_base_of_v<panel_mat_base, mat_type> && !std::is_base_of_v<panel_mat_base, cache_type>)
    void eval_then_add_to(mat_type &dst, cache_type &cache) const {
        assert(dst.rows() >= row_ed_ && dst.cols() >= col_ed_ && "output matrix size mismatch");
        auto out_block = dst.block(row_st_, col_st_, this->rows_, this->cols_);
        if constexpr (config.overwrite)
            out_block.diagonal().setConstant(1.0);
        else
            out_block.diagonal().array() += 1.0; // add ones to the diagonal
    }
};

template <typename T, bool is_rhs, bool _transposed>
struct operand {
    static constexpr bool transposed = _transposed;
    static constexpr bool is_panel = std::is_base_of_v<panel_mat_base, T>;
    static constexpr bool is_eigen_expr = is_sp_expr<T>::value;
    static constexpr auto is_diag = is_diag_type<T>::value;

    static constexpr auto get_sp_type() {
        if constexpr (is_panel) {
            return T::sparsity_type;
        } else if constexpr (is_diag) {
            return sparsity::diag;
        } else {
            return sparsity::dense;
        }
    }
    static constexpr auto sp_type = get_sp_type();
    static constexpr bool is_eye = sp_type == sparsity::eye;

    const T &val;
    int inner_st, inner_dim, inner_ed;
    int outer_st, outer_dim, outer_ed;
    int st, dim;
    template <bool flip = false>
    auto op_inner_info() {
        if constexpr (is_panel) {
            if constexpr ((is_rhs == transposed) != flip) {
                return std::tuple{val.col_st_, val.cols_, val.col_ed_};
            } else {
                return std::tuple{val.row_st_, val.rows_, val.row_ed_};
            }
        } else {
            if constexpr ((is_rhs == transposed) != flip) {
                return std::tuple{0, val.cols(), val.cols()};
            } else {
                return std::tuple{0, val.rows(), val.rows()};
            }
        }
    }
    auto op_outer_info() {
        return op_inner_info<true>();
    }
    operand(const T &val) : val(val) {
        std::tie(inner_st, inner_dim, inner_ed) = op_inner_info();
        std::tie(outer_st, outer_dim, outer_ed) = op_outer_info();
    }
    int clip_outer_dim(int clip_res, int clip_val) const {
        if constexpr (sp_type == sparsity::dense) {
            return clip_val;
        } else {
            return std::max<int>(std::min<int>(clip_res + clip_val, outer_ed), outer_st);
        }
    }
    decltype(auto) get_impl() const {
        constexpr bool col_block = is_rhs == transposed; // true if we are dealing with a column block
        if constexpr (is_panel && !is_eigen_expr) {
            if constexpr (sp_type == sparsity::dense) {
                if constexpr (col_block) {
                    return val.data_.middleCols(st, dim); // select the relevant columns from rhs expression
                } else {
                    return val.data_.middleRows(st, dim); // select the relevant rows from lhs expression
                }
            } else if constexpr (sp_type == sparsity::diag) {
                return val.data_.segment(st, dim);
            } else if constexpr (sp_type == sparsity::eye) {
                return non_op{dim};
            }
        } else {
            if constexpr (is_eye) {
                return non_op{dim};
            } else if constexpr (is_diag) {
                return val.diagonal().segment(st, dim);
            } else {
                if constexpr (col_block) {
                    return val.middleCols(st, dim); // select the relevant columns from rhs expression
                } else {
                    return val.middleRows(st, dim); // select the relevant rows from rhs expression
                }
            }
        }
    }

    decltype(auto) get() const {
        if constexpr (transposed && !is_diag) {
            return get_impl().transpose();
        } else {
            return get_impl();
        }
    }

    template <typename rhs_type>
        requires(is_rhs == false)
    decltype(auto) apply_this_on_left(const rhs_type &_rhs) {
        auto lhs = get();
        auto rhs = _rhs.get();
        if constexpr (sp_type == sparsity::dense) {
            if constexpr (rhs_type::is_eye) {
                return lhs;
            } else if constexpr (rhs_type::is_diag) {
                return lhs * rhs.asDiagonal();
            } else {
                return lhs * rhs;
            }
        } else if constexpr (sp_type == sparsity::diag) {
            if constexpr (rhs_type::is_eye) {
                return lhs.asDiagonal();
            } else if constexpr (rhs_type::is_diag) {
                return lhs.cwiseProduct(rhs).asDiagonal();
            } else {
                return lhs.asDiagonal() * rhs;
            }
        } else if constexpr (sp_type == sparsity::eye) {
            if constexpr (rhs_type::is_eye) {
                return lhs;
            } else if constexpr (rhs_type::is_diag) {
                return rhs.asDiagonal();
            } else {
                return rhs;
            }
        }
    }
};

template <bool lhs_transposed, bool rhs_transposed, typename lhs_type, typename rhs_type>
struct binary_op {
    int rows_;
    int cols_;
    int row_st_;
    int col_st_;

    operand<lhs_type, false, lhs_transposed> lhs_; // panel matrix
    operand<rhs_type, true, rhs_transposed> rhs_;  // right-hand side expression

    binary_op(const lhs_type &lhs, const rhs_type &rhs) : lhs_(lhs), rhs_(rhs) {
        int inner_idx_st = std::max<int>(lhs_.inner_st, rhs_.inner_st);                  // intersection of rows - starting point
        int inner_idx_ed = std::min<int>(lhs_.inner_ed, rhs_.inner_ed);                  // intersection of rows - ending point
        int dim_mid = std::max<int>(inner_idx_ed - inner_idx_st, 0);                     // number of intersecting rows
        rhs_.st = std::min<int>(inner_idx_st - rhs_.inner_st, rhs_.inner_dim - dim_mid); // relative starting point in rhs
        lhs_.st = std::min<int>(inner_idx_st - lhs_.inner_st, lhs_.inner_dim - dim_mid); // relative starting point in lhs
        row_st_ = lhs_.clip_outer_dim(rhs_.inner_st - lhs_.inner_st, lhs_.outer_st);     // output info for the resulting expression
        col_st_ = rhs_.clip_outer_dim(lhs_.inner_st - rhs_.inner_st, rhs_.outer_st);     // output info for the resulting expression
        lhs_.dim = dim_mid;
        rhs_.dim = dim_mid;
        rows_ = lhs_.clip_outer_dim(inner_idx_st + dim_mid - lhs_.inner_ed, lhs_.outer_ed) - row_st_;
        cols_ = rhs_.clip_outer_dim(inner_idx_st + dim_mid - rhs_.inner_ed, rhs_.outer_ed) - col_st_; // number of intersecting columns
    }

    bool valid() const { return rows_ > 0 && cols_ > 0; }

    decltype(auto) run() {
        return retval(lhs_.apply_this_on_left(rhs_));
    }

    decltype(auto) retval(auto &&ret) {
        if constexpr (std::is_same_v<std::decay_t<decltype(ret)>, non_op>) {
            assert(ret.dim == rows_ && ret.dim == cols_ && "resulting expression size mismatch");
        } else
            assert(ret.rows() == rows_ && ret.cols() == cols_ && "resulting expression size mismatch");
        return sp_expr{std::move(ret), std::make_pair(row_st_, col_st_)};
    }
};

template <typename lhs_type, typename rhs_type>
using spmm_lt_rn = binary_op<true, false, lhs_type, rhs_type>;

template <typename lhs_type, typename rhs_type>
using spmm_ln_rn = binary_op<false, false, lhs_type, rhs_type>;

template <typename lhs_type, typename rhs_type>
using spmm_lt_rt = binary_op<true, true, lhs_type, rhs_type>;

template <typename lhs_type, typename rhs_type>
using spmm_ln_rt = binary_op<false, true, lhs_type, rhs_type>;

} // namespace moto

#endif