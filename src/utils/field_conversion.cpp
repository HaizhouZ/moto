#include <moto/ocp/problem.hpp>
#include <moto/ocp/sym.hpp>
#include <moto/utils/field_conversion.hpp>

namespace moto {
namespace utils {

void copy_y_to_x(vector_ref from_y, vector_ref to_x,
                 const ocp *prob_y, const ocp *prob_x) {
    for (expr &y : prob_y->exprs(__y)) {
        prob_x->extract(to_x, static_cast<sym &>(y).prev()) = prob_y->extract(from_y, y);
    }
}
/**
 * @brief forward copy from stacked x to y
 */
void copy_x_to_y(vector_ref from_x, vector_ref to_y,
                 const ocp *prob_x, const ocp *prob_y) {
    for (expr &x : prob_x->exprs(__x)) {
        prob_y->extract(to_y, static_cast<sym &>(x).next()) = prob_x->extract(from_x, x);
    }
}

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
        for (sym &x : prob_x->exprs(__x)) {
            auto &y = x.next();
            size_t x0 = prob_x->get_expr_start(x);
            size_t x1 = x0 + x.dim();
            size_t y0 = prob_y->get_expr_start(y);
            for (size_t i : range(x0, x1)) {
                perm.indices()[y0++] = i;
            }
        }
        return perm;
    }
}

} // namespace utils
} // namespace moto