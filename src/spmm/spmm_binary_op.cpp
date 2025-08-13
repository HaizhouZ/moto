#include <moto/spmm/impl/buffer.hpp>
#include <moto/spmm/impl/spmm_impl.hpp>
#include <moto/spmm/sparse_mat.hpp>

#include <magic_enum/magic_enum_utility.hpp>

namespace moto {

template <sparsity Sp, typename Callback>
auto unary_select(const sparse_mat &s, Callback &&callback) {
    if constexpr (Sp == sparsity::dense) {
        for (const auto &panel : s.dense_panels_) {
            callback(panel);
        }
    } else if constexpr (Sp == sparsity::diag) {
        for (const auto &panel : s.diag_panels_) {
            callback(panel);
        }
    } else if constexpr (Sp == sparsity::eye) {
        for (const auto &panel : s.eye_panels_) {
            callback(panel);
        }
    } else {
    }
};

template <bool ltr, bool rtr, bool add, typename lhs_type, typename rhs_type, typename out_type>
void product(const lhs_type &lhs, const rhs_type &rhs, out_type &out) {
    constexpr auto l_is_sp_mat = std::is_same_v<std::decay_t<lhs_type>, sparse_mat>;
    constexpr auto r_is_sp_mat = std::is_same_v<std::decay_t<rhs_type>, sparse_mat>;
    auto product_impl = [&out]<typename L_, typename R_>(const L_ &lhs, const R_ &rhs) {
        thread_local spmm::buffer cache_lhs;
        auto lhs_expr = binary_op<ltr, rtr, L_, R_>(lhs, rhs);
        if (lhs_expr.valid()) {
            auto lhs_res = lhs_expr.run();
            cache_lhs.resize(lhs_res);
            constexpr auto config = eval_config{.add_to = add};
            lhs_res.template eval_then_add_to(out, cache_lhs.data_);
        }
    };
    if constexpr (l_is_sp_mat && r_is_sp_mat) {
        constexpr size_t num = static_cast<size_t>(sparsity::num);

        magic_enum::enum_for_each<sparsity>([&](auto lsp) {
            unary_select<lsp>(lhs, [&](auto &_lhs) {
                magic_enum::enum_for_each<sparsity>([&](auto rsp) {
                    unary_select<rsp>(rhs, [&](auto &_rhs) {
                        product_impl(_lhs, _rhs);
                    });
                });
            });
        });
    } else if constexpr (l_is_sp_mat) {
        magic_enum::enum_for_each<sparsity>([&](auto lsp) {
            unary_select<lsp>(lhs, [&](auto &_lhs) {
                product_impl(_lhs, rhs);
            });
        });
    } else if constexpr (r_is_sp_mat) {
        magic_enum::enum_for_each<sparsity>([&](auto rsp) {
            unary_select<rsp>(rhs, [&](auto &_rhs) {
                product_impl(lhs, _rhs);
            });
        });
    } else {
        static_assert(false, "Unsupported combination of lhs and rhs types for product");
    }
}

template <typename rhs_type, typename out_type, bool add>
void sparse_mat::times(const rhs_type &rhs, out_type &out) {
    product<false, false, add>(*this, rhs, out);
}

template <typename rhs_type, typename out_type, bool add>
void sparse_mat::T_times(const rhs_type &rhs, out_type &out) {
    product<true, false, add>(*this, rhs, out);
}
template <typename lhs_type, typename out_type, bool add>
void sparse_mat::right_times(const lhs_type &lhs, out_type &out) {
    product<false, false, add>(lhs, *this, out);
}

template <typename lhs_type, typename out_type, bool add>
void sparse_mat::right_T_times(const lhs_type &lhs, out_type &out) {
    product<true, false, add>(lhs, *this, out);
}

#define EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, lhs_type, rhs_type)                            \
    template void sparse_mat::func<lhs_type, rhs_type, true>(const lhs_type &rhs, rhs_type &out); \
    template void sparse_mat::func<lhs_type, rhs_type, false>(const lhs_type &rhs, rhs_type &out);

#define EXPLICIT_SP_MEMFUNC_INSTANTIATE(func)                       \
    EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, sparse_mat, matrix); \
    EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, sparse_mat, vector); \
    EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, matrix, matrix);     \
    EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, matrix, vector);     \
    EXPLICIT_SP_MEMFUNC_INSTANTIATE_IMPL(func, vector, vector);

EXPLICIT_SP_MEMFUNC_INSTANTIATE(times)
EXPLICIT_SP_MEMFUNC_INSTANTIATE(T_times)
EXPLICIT_SP_MEMFUNC_INSTANTIATE(right_times)
EXPLICIT_SP_MEMFUNC_INSTANTIATE(right_T_times)

} // namespace moto