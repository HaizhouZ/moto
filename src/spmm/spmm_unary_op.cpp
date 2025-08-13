
#include <moto/spmm/sparse_mat.hpp>

#include <moto/spmm/impl/spmm_impl.hpp>

#include <moto/spmm/impl/buffer.hpp>

namespace moto {

void sparse_mat::inner_product(const matrix &m, matrix &out) {
    using buffer = spmm::buffer;
    thread_local buffer cache_rhs;
    assert(m.rows() == m.cols() && m.cols() == rows_ && "matrix size mismatch");

    auto inner_product_lhs_run =
        [&]<bool same_sides = false, typename L_, typename R_>(const L_ &lhs, const R_ &rhs, std::bool_constant<same_sides> = {}) {
            thread_local buffer cache_lhs;
            auto lhs_expr = spmm_lt_rn(lhs, rhs);
            if (lhs_expr.valid()) {
                auto lhs_res = lhs_expr.run();
                cache_lhs.resize(lhs_res);
                constexpr auto config = eval_config{.sym_res = !same_sides, .same_sides = same_sides, .dst_panel = false, .overwrite = false};
                lhs_res.template eval_then_add_to<config>(out, cache_lhs.data_);
            }
        };

    auto inner_product_rhs_run = [&]<typename R_>(R_ &&rhs) {
        auto rhs_res = rhs.run();
        cache_rhs.resize(rhs_res);
        constexpr auto config = eval_config{.sym_res = false, .same_sides = false, .dst_panel = true, .overwrite = true};
        rhs_res.template eval_then_add_to<config>(cache_rhs.data_, cache_rhs.null_); // cache the result
    };

    // firstly match all eye panels because the rhs is only cropping the matrix
    for (size_t r_idx = 0; r_idx < eye_panels_.size(); r_idx++) {
        auto &rhs = eye_panels_[r_idx];
        auto rhs_expr = spmm_ln_rn(m, rhs);
        if (rhs_expr.valid()) { // firstly compute inner product of the eye panel, no cache requried
            auto rhs_res = rhs_expr.run();
            inner_product_lhs_run(rhs, rhs_res, std::bool_constant<true>{}); // run the eye panel with the cached result
            for (size_t l_idx = r_idx + 1; l_idx < eye_panels_.size(); ++l_idx) {
                auto &lhs = eye_panels_[l_idx];
                inner_product_lhs_run(lhs, rhs_res); // run the diag panel with the cached result
            }
        }
    }
    // then match all diag panels, cropping the matrix and multiplying by the diagonal
    for (size_t r_idx = 0; r_idx < diag_panels_.size(); r_idx++) {
        auto &rhs = diag_panels_[r_idx];
        auto rhs_expr = spmm_ln_rn(m, rhs);
        if (rhs_expr.valid()) {
            inner_product_rhs_run(rhs_expr); // cache the result of D * diag panel

            inner_product_lhs_run(rhs, cache_rhs, std::bool_constant<true>{}); // run the diag panel with the cached result
            for (auto &lhs : eye_panels_) {
                inner_product_lhs_run(lhs, cache_rhs); // run the diag panel with the cached result
            }
            for (size_t l_idx = r_idx + 1; l_idx < diag_panels_.size(); ++l_idx) {
                auto &lhs = diag_panels_[l_idx];
                inner_product_lhs_run(lhs, cache_rhs); // run the diag panel with the cached result
            }
        }
    }
    // finally match all dense panels
    for (size_t r_idx = 0; r_idx < dense_panels_.size(); r_idx++) {
        size_t lhs_st_min, lhs_ed_max;
        auto &rhs = dense_panels_[r_idx];
        auto rhs_expr = spmm_ln_rn(m, rhs);
        if (rhs_expr.valid()) {
            inner_product_rhs_run(rhs_expr); // cache the result of D * dense panel

            inner_product_lhs_run(rhs, cache_rhs, std::bool_constant<true>{}); // run the dense panel with the cached result

            for (auto &lhs : eye_panels_) {
                inner_product_lhs_run(lhs, cache_rhs);
            }
            for (auto &lhs : diag_panels_) {
                inner_product_lhs_run(lhs, cache_rhs);
            }
            for (size_t l_idx = r_idx + 1; l_idx < dense_panels_.size(); ++l_idx) {
                auto &lhs = dense_panels_[l_idx];
                inner_product_lhs_run(lhs, cache_rhs);
            }
        }
    }
}

} // namespace moto