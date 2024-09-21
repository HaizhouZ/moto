#ifndef __MULTIVARIATE__
#define __MULTIVARIATE__

#include <manbo/core/expression_base.hpp>
#include <manbo/core/data_types.hpp>
#include <functional>

namespace manbo {

class multivariate : public expr_base {
   protected:
    const approx_type level_ = approx_type::zero;

   public:
    typedef std::vector<const scalar_t*> stacked_const_ptr_t;

    approx_type approx_level() { return level_; }

    multivariate(const std::string& name, size_t dim, field_type field, approx_type approx)
        : expr_base(name, dim, field), level_(approx) {}

    void add_in(sym_ptr_t sym) {
        in_.emplace_back(sym);
    }
    void add_in(const std::vector<sym_ptr_t>& syms) {
        in_.insert(in_.end(), syms.begin(), syms.end());
    }
    void get_primal(std::shared_ptr<problem_t> problem,
                    primal_data_ptr_t raw,
                    stacked_const_ptr_t& in) {
        for (size_t i = 0; i < in_.size(); i++) {
            auto sym = in_[i];
            in[i] = problem->get_data_ptr(raw[sym->field_], sym);
        }
    }
    size_t n_in() { return in_.size(); }
    // dst bind to std::function
    // call to
    // constr1(q)
    // constr2(q2)

   protected:
    std::vector<sym_ptr_t> in_;
};
}  // namespace manbo

#endif /*__group_eval_*/