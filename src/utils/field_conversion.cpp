#include <moto/ocp/impl/func.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/utils/field_conversion.hpp>
#include <ranges>

namespace moto {
namespace utils {

void copy_y_to_x(vector_ref from_y, vector_ref to_x,
                 const ocp *prob_y, const ocp *prob_x) {
    // for (sym &y : prob_y->exprs(__y)) {
    //     prob_x->extract(to_x, y.prev()) = prob_y->extract(from_y, y);
    // }
    assert(prob_y->num(__dyn) == prob_x->num(__dyn) && "number of dynamics expressions must match");
    auto &dyn_y = prob_y->exprs(__dyn);
    for (size_t i = 0; i < dyn_y.size(); ++i) {
        const generic_func &y = dyn_y[i];
        to_x.segment(prob_x->get_expr_start(y.in_args(__y)[0]->prev()), y.arg_dim(__x)) =
            from_y.segment(prob_y->get_expr_start(y), y.arg_dim(__y));
    }
}
/**
 * @brief forward copy from stacked x to y
 */
void copy_x_to_y(vector_ref from_x, vector_ref to_y,
                 const ocp *prob_x, const ocp *prob_y) {
    // for (sym &x : prob_x->exprs(__x)) {
    //     prob_y->extract(to_y, x.next()) = prob_x->extract(from_x, x);
    // }
    assert(prob_x->num(__dyn) == prob_y->num(__dyn) && "number of dynamics expressions must match");
    auto &dyn_x = prob_x->exprs(__dyn);
    for (size_t i = 0; i < dyn_x.size(); ++i) {
        const generic_func &x = dyn_x[i];
        to_y.segment(prob_y->get_expr_start(x.in_args(__x)[0]->next()), x.arg_dim(__x)) =
            from_x.segment(prob_x->get_expr_start(x), x.arg_dim(__x));
    }
}

/// @todo change to block permutation
Eigen::PermutationMatrix<-1, -1> &permutation_from_y_to_x(const ocp *prob_y, const ocp *prob_x) {
    using perm_type = Eigen::PermutationMatrix<-1, -1>;
    static std::unordered_map<size_t, std::unordered_map<size_t, perm_type>> perm_cache;
    assert(prob_y->dim(__y) == prob_x->dim(__x) && "dimension between states must match!");
    perm_cache.try_emplace(prob_y->uid());
    auto [it, inserted] = perm_cache[prob_y->uid()].try_emplace(prob_x->uid(), prob_x->dim(__x));
    if (!inserted)
        return it->second; // already exists
    else {
        auto &perm = it->second;
        size_t col_y = 0;
        for (sym &y : prob_y->exprs(__y)) {
            sym &x = y.prev();
            size_t x0 = prob_x->get_expr_start(x);
            size_t x1 = x0 + x.dim();
            for (size_t i : range(x0, x1)) {
                perm.indices()[col_y++] = i;
            }
        }
        return perm;
    }
}

} // namespace utils
} // namespace moto