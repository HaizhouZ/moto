
#include <moto/spmm/sparse_mat.hpp>

#include <moto/spmm/impl/spmm_impl.hpp>

#include <moto/spmm/impl/buffer.hpp>

namespace moto {
void sparse_mat::inner_product(const matrix &m, matrix &out) {
    using buffer = spmm::buffer;
    thread_local buffer cache_rhs;
    assert(m.rows() == m.cols() && m.cols() == rows_ && "matrix size mismatch");
    // cache_rhs.resize(cols_, rows_);
    cache_rhs.resize(rows_, cols_);
    cache_rhs.data_.setZero(); // ensure the cache is zeroed out
    // T_times(m, cache_rhs.data_); // run the inner product
    // right_times(cache_rhs.data_, out); // cache the result
    right_times(m, cache_rhs.data_); // cache the result
    T_times(cache_rhs.data_, out); // run the inner product
}
// void sparse_mat::inner_product(const matrix &m, matrix &out) {
//     using buffer = spmm::buffer;
//     thread_local buffer cache_rhs;
//     assert(m.rows() == m.cols() && m.cols() == rows_ && "matrix size mismatch");

//     auto inner_product_lhs_run =
//         [&]<bool same_sides = false, typename L_, typename R_>(const L_ &lhs, const R_ &rhs, std::bool_constant<same_sides> = {}) {
//             thread_local buffer cache_lhs;
//             auto lhs_expr = spmm::binary_op<true, false, true, false, L_, R_>(lhs, rhs);
//             if (lhs_expr.valid()) {
//                 auto lhs_res = lhs_expr.run();
//                 // if constexpr (!same_sides)
//                 //     cache_lhs.resize(lhs_res);
//                 constexpr auto config = spmm::eval_config{.sym_res = true, .same_sides = same_sides};
//                 lhs_res.template eval_then_add_to<config>(out, cache_lhs.data_);
//             }
//         };

//     auto eval_rhs_to_cache = [&](auto &&rhs_res) {
//         cache_rhs.resize(rhs_res);
//         constexpr auto config = spmm::eval_config{.no_offset = true, .overwrite = true};
//         rhs_res.template eval_then_add_to<config>(cache_rhs.data_, cache_rhs.null_); // cache the result
//     };

//     auto run_and_eval_rhs_to_cache = [&]<typename R_>(R_ &&rhs) {
//         auto rhs_res = rhs.run();
//         eval_rhs_to_cache(rhs_res);
//     };

//     // firstly match all eye panels because the rhs is only cropping the matrix
//     for (size_t r_idx = 0; r_idx < eye_panels_.size(); r_idx++) {
//         auto &rhs = eye_panels_[r_idx];
//         auto rhs_expr = spmm::product_ln_rn(m, rhs);
//         if (rhs_expr.valid()) { // firstly compute inner product of the eye panel, no cache requried
//             auto rhs_res = rhs_expr.run();
//             inner_product_lhs_run(rhs, rhs_res, std::bool_constant<true>{}); // run the eye panel with the cached result
//             for (size_t l_idx = r_idx + 1; l_idx < eye_panels_.size(); ++l_idx) {
//                 auto &lhs = eye_panels_[l_idx];
//                 inner_product_lhs_run(lhs, rhs_res); // run the diag panel with the cached result
//             }
//         }
//     }
//     // then match all diag panels, cropping the matrix and multiplying by the diagonal
//     for (size_t r_idx = 0; r_idx < diag_panels_.size(); r_idx++) {
//         auto &rhs = diag_panels_[r_idx];
//         // auto rhs_expr = spmm::product_ln_rn(m, rhs);
//         auto rhs_expr = spmm::binary_op<false, false, false, true, std::decay_t<decltype(m)>, std::decay_t<decltype(rhs)>>(m, rhs);
//         if (rhs_expr.valid()) {
//             auto rhs_res = rhs_expr.run();
//             // int lhs_inner_st_min = rhs.row_st_, lhs_inner_ed_max = rhs.row_ed_;
//             // bool lhs_valid = false;
//             // auto clip_rhs_rows = [&](const auto &lhs) {
//             //     auto lhs_expr = spmm::product_lt_rn(lhs, rhs_res);
//             //     if (lhs_expr.valid()) {
//             //         lhs_valid = true;
//             //         lhs_inner_st_min = std::min(lhs_inner_st_min, lhs_expr.lhs_.inner_st + lhs_expr.lhs_.st);
//             //         lhs_inner_ed_max = std::max(lhs_inner_ed_max, lhs_expr.lhs_.inner_st + lhs_expr.lhs_.st + lhs_expr.lhs_.dim);
//             //     }
//             // };
//             // for (auto &lhs : eye_panels_) {
//             //     clip_rhs_rows(lhs);
//             // }
//             // for (size_t l_idx = r_idx + 1; l_idx < diag_panels_.size(); ++l_idx) {
//             //     auto &lhs = diag_panels_[l_idx];
//             //     clip_rhs_rows(lhs);
//             // }
//             // if (lhs_valid) {
//             // if (lhs_inner_ed_max == rhs_res.row_ed_ && lhs_inner_st_min == rhs_res.row_st_) {
//             //     // if the lhs is not cropped, we can run the dense panel directly
//             //     eval_rhs_to_cache(rhs_res); // run the dense panel with the cached result
//             // } else {
//             //     // otherwise we need to crop the rhs and run the dense panel
//             //     assert(lhs_inner_st_min < lhs_inner_ed_max && "lhs inner range is invalid");
//             //     auto cropped_rhs = spmm::sp_expr(rhs_res.middleRows(lhs_inner_st_min, lhs_inner_ed_max - lhs_inner_st_min),
//             //                                      {lhs_inner_st_min, rhs_res.col_st_});
//             //     eval_rhs_to_cache(cropped_rhs); // run the dense panel with the cropped result
//             // }
//             inner_product_lhs_run(rhs, rhs_res, std::bool_constant<true>{}); // run the dense panel with the cached result

//             for (auto &lhs : eye_panels_) {
//                 inner_product_lhs_run(lhs, rhs_res);
//             }
//             for (size_t l_idx = 0; l_idx < diag_panels_.size(); ++l_idx) {
//                 if (l_idx == r_idx)
//                     continue;
//                 auto &lhs = diag_panels_[l_idx];
//                 inner_product_lhs_run(lhs, rhs_res);
//             }
//             // } else {
//             //     inner_product_lhs_run(rhs, rhs_res, std::bool_constant<true>{}); // run the dense panel with the cached result
//             // }
//         }
//     }

//     // finally match all dense panels
//     // for (size_t r_idx = 0; r_idx < dense_panels_.size(); r_idx++) {
//     //     auto &rhs = dense_panels_[r_idx];
//     //     // auto rhs_expr = spmm::product_ln_rn(m, rhs);
//     //     auto rhs_expr = spmm::binary_op<false, false, false, true, std::decay_t<decltype(m)>, std::decay_t<decltype(rhs)>>(m, rhs);
//     //     if (rhs_expr.valid()) { /// @todo : clip the rhs cache by lhs (?)
//     //         auto rhs_res = rhs_expr.run();
//     //         // int lhs_inner_st_min = rhs.row_st_, lhs_inner_ed_max = rhs.row_ed_;
//     //         // bool lhs_valid = false;
//     //         // auto clip_rhs_rows = [&](const auto &lhs) {
//     //         //     auto lhs_expr = spmm::product_lt_rn(lhs, rhs_res);
//     //         //     if (lhs_expr.valid()) {
//     //         //         lhs_valid = true;
//     //         //         lhs_inner_st_min = std::min(lhs_inner_st_min, lhs_expr.lhs_.inner_st + lhs_expr.lhs_.st);
//     //         //         lhs_inner_ed_max = std::max(lhs_inner_ed_max, lhs_expr.lhs_.inner_st + lhs_expr.lhs_.st + lhs_expr.lhs_.dim);
//     //         //     }
//     //         // };
//     //         // for (auto &lhs : eye_panels_) {
//     //         //     clip_rhs_rows(lhs);
//     //         // }
//     //         // for (auto &lhs : diag_panels_) {
//     //         //     clip_rhs_rows(lhs);
//     //         // }
//     //         // for (size_t l_idx = r_idx + 1; l_idx < dense_panels_.size(); ++l_idx) {
//     //         //     auto &lhs = dense_panels_[l_idx];
//     //         //     clip_rhs_rows(lhs);
//     //         // }
//     //         // if (lhs_valid) {
//     //         // if (lhs_inner_ed_max == rhs_res.row_ed_ && lhs_inner_st_min == rhs_res.row_st_) {
//     //         //     // if the lhs is not cropped, we can run the dense panel directly
//     //         //     eval_rhs_to_cache(rhs_res); // run the dense panel with the cached result
//     //         // } else {
//     //         //     // otherwise we need to crop the rhs and run the dense panel
//     //         //     assert(lhs_inner_st_min < lhs_inner_ed_max && "lhs inner range is invalid");
//     //         //     auto cropped_rhs = spmm::sp_expr(rhs_res.middleRows(lhs_inner_st_min, lhs_inner_ed_max - lhs_inner_st_min),
//     //         //                                      {lhs_inner_st_min, rhs_res.col_st_});
//     //         //     eval_rhs_to_cache(cropped_rhs); // run the dense panel with the cropped result
//     //         // }
//     //         inner_product_lhs_run(rhs, rhs_res, std::bool_constant<true>{}); // run the dense panel with the cached result

//     //         for (auto &lhs : eye_panels_) {
//     //             inner_product_lhs_run(lhs, rhs_res);
//     //         }
//     //         for (auto &lhs : diag_panels_) {
//     //             inner_product_lhs_run(lhs, rhs_res);
//     //         }
//     //         for (size_t l_idx = 0; l_idx < dense_panels_.size(); ++l_idx) {
//     //             if (l_idx == r_idx)
//     //                 continue;
//     //             auto &lhs = dense_panels_[l_idx];
//     //             inner_product_lhs_run(lhs, rhs_res);
//     //         }
//     //         // } else {
//     //         //     inner_product_lhs_run(rhs, rhs_res, std::bool_constant<true>{}); // run the dense panel with the cached result
//     //         // }
//     //     }
//     // }
// }

} // namespace moto