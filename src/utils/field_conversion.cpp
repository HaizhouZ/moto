#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/utils/field_conversion.hpp>
#include <ranges>
#include <unordered_map>

namespace moto {
namespace utils {

void copy_y_to_x(vector_ref from_y, vector_ref to_x,
                 const ocp *prob_y, const ocp *prob_x) {
    for (const sym &y : prob_y->exprs(__y)) {
        const sym &x = *y.prev();
        if (!prob_x->is_active(x))
            throw std::runtime_error(fmt::format(
                "next stage is missing active state {} uid {}", x.name(), x.uid()));
        prob_x->extract(to_x, x) = prob_y->extract(from_y, y);
    }
}
void copy_y_to_x_tangent(vector_ref from_y, vector_ref to_x,
                         const ocp *prob_y, const ocp *prob_x) {
    for (const sym &y : prob_y->exprs(__y)) {
        const sym &x = *y.prev();
        if (!prob_x->is_active(x))
            throw std::runtime_error(fmt::format(
                "next stage is missing active state {} uid {}", x.name(), x.uid()));
        prob_x->extract_tangent(to_x, x) = prob_y->extract_tangent(from_y, y);
    }
}
/**
 * @brief forward copy from stacked x to y
 */
void copy_x_to_y(vector_ref from_x, vector_ref to_y,
                 const ocp *prob_x, const ocp *prob_y) {
    for (const sym &x : prob_x->exprs(__x)) {
        const sym &y = *x.next();
        if (!prob_y->is_active(y))
            throw std::runtime_error(fmt::format(
                "previous stage is missing active next-state {} uid {}",
                y.name(), y.uid()));
        prob_y->extract(to_y, y) = prob_x->extract(from_x, x);
    }
}
void copy_x_to_y_tangent(vector_ref from_x, vector_ref to_y,
                         const ocp *prob_x, const ocp *prob_y) {
    for (const sym &x : prob_x->exprs(__x)) {
        const sym &y = *x.next();
        if (!prob_y->is_active(y))
            throw std::runtime_error(fmt::format(
                "previous stage is missing active next-state {} uid {}",
                y.name(), y.uid()));
        prob_y->extract_tangent(to_y, y) = prob_x->extract_tangent(from_x, x);
    }
}

/// @todo change to block permutation
Eigen::PermutationMatrix<-1, -1> &permutation_from_y_to_x(const ocp *prob_y, const ocp *prob_x) {
    using perm_type = Eigen::PermutationMatrix<-1, -1>;
    thread_local std::unordered_map<size_t, std::unordered_map<size_t, perm_type>> perm_cache;
    if (prob_y->tdim(__y) != prob_x->tdim(__x))
        throw std::runtime_error("tangent space dimension between adjacent states must match");
    auto &by_x = perm_cache.try_emplace(prob_y->uid()).first->second;
    auto [it, inserted] = by_x.try_emplace(prob_x->uid(), prob_x->tdim(__x));
    if (!inserted)
        return it->second; // already exists
    else {
        auto &perm = it->second;
        size_t col_y = 0;
        for (sym &y : prob_y->exprs(__y)) {
            sym &x = y.prev();
            if (!prob_x->is_active(x))
                throw std::runtime_error(fmt::format(
                    "next stage is missing active state {} uid {}", x.name(), x.uid()));
            size_t x0 = prob_x->get_expr_start_tangent(x);
            size_t x1 = x0 + x.tdim();
            for (size_t i : range(x0, x1)) {
                perm.indices()[col_y++] = i;
            }
        }
        return perm;
    }
}

} // namespace utils
} // namespace moto
