#ifndef __MULTIVARIATE__
#define __MULTIVARIATE__

#include <atri/core/expression_base.hpp>
#include <atri/core/data_types.hpp>
#include <functional>

namespace atri {

class multivariate : public expr {
   protected:
    const approx_type level_ = approx_type::zero;

   public:
    typedef std::vector<const scalar_t*> stacked_const_ptr_t;

    approx_type approx_level() { return level_; }

    multivariate(const std::string& name, size_t dim, field_type field, approx_type approx)
        : expr(name, dim, field), level_(approx) {}

    void add_in(sym_ptr_t sym) {
        in_idx_[sym->uid_] = in_.size();
        in_.push_back(sym);
    }
    void add_in(const std::vector<sym_ptr_t>& syms) {
        for (auto s : syms) {
            add_in(s);
        }
    }
    void get_primal(std::shared_ptr<problem> problem,
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
    std::map<size_t, size_t> in_idx_;
};
def_ptr(multivariate);
}  // namespace atri

#endif /*__group_eval_*/