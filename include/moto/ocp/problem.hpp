#ifndef __MOTO_PROBLEM_HPP__
#define __MOTO_PROBLEM_HPP__

#include <array>
#include <map>
#include <unordered_map>
#include <vector>

#include <moto/core/expr.hpp>

namespace moto {
/**
 * @brief problem formulation of an OCP stage
 *
 */
class ocp {
  protected:
    ocp() { uid_.set_inc(); }; // default constructor
    ocp(const ocp &rhs) = default; // copy constructor
    bool add_impl(expr &);
    void maintain_order(expr &);
    static size_t max_uid; ///< uid used to index global expressions
    utils::unique_id<ocp> uid_;
    /// collection of all expressions in the problem
    std::array<expr_list, field::num> expr_;
    /// data index of expr in serialized vector, by uid
    std::unordered_map<size_t, size_t> d_idx_;
    /// data index of tagent space var of expr in serialized vector, by uid
    std::unordered_map<size_t, size_t> d_idx_tangent_;
    /// position of expr in expr_ e.g., no. xxx, by uid
    std::unordered_map<size_t, size_t> pos_by_uid_;
    /// dimension of each field
    std::array<size_t, field::num> dim_{};
    /// tangent space dimension of each field
    std::array<size_t, field::num_prim> tdim_{};

  public:
    CONST_PROPERTY(uid);                                                       ///< getter for uid
    const auto &exprs(size_t f) const { return expr_.at(f); }                  ///< getter for expr_
    const auto &pos(const expr &ex) const { return pos_by_uid_.at(ex.uid()); } ///< getter for pos_by_uid_
    size_t dim(size_t f) const { return dim_.at(f); }                          ///< getter for dim_
    size_t num(size_t f) const { return expr_[f].size(); } ///< getter for num of exprs in field f
    size_t tdim(size_t f) const { return tdim_.at(f); }                        ///< getter for tdim_
    void wait_until_ready() const;

    void print_summary() const;

    static auto create() { return std::shared_ptr<ocp>(new ocp()); }
    auto clone() { return std::shared_ptr<ocp>(new ocp(*this)); }

    scalar_t *get_data_ptr(scalar_t *data, const expr &ex) const {
        return data + get_expr_start(ex);
    }
    scalar_t *get_data_ptr(scalar_t *data, const expr &ex, size_t offset) const {
        return data + get_expr_start(ex) * offset;
    }
    vector_ref extract(vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start(ex), ex.dim());
    }
    vector_ref extract_tangent(vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start_tangent(ex), ex.tdim());
    }
    row_vector_ref extract_row(row_vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start(ex), ex.dim());
    }
    row_vector_ref extract_row_tangent(row_vector_ref data, const expr &ex) const {
        return data.segment(get_expr_start_tangent(ex), ex.tdim());
    }
    /**
     * @brief add expr to problem formulation
     * @note will copy the shared pointer
     * @param ex expression to be added
     */
    template <typename T>
        requires std::is_base_of_v<shared_expr, std::remove_cvref_t<T>> ||
                 std::is_base_of_v<expr, std::remove_reference_t<T>>
    void add(T &&ex) {
        if (add_impl(ex)) {
            expr& ex_ = expr_[static_cast<const expr &>(ex).field()].emplace_back(std::forward<T>(ex));
            maintain_order(ex_);
        }
    }

    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }

    /**
     * @brief get start index of expr in its field
     */
    size_t get_expr_start(const expr &ex) const {
        try {
            return d_idx_.at(ex.uid());
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        }
    }

    size_t get_expr_start_tangent(const expr &ex) const {
        try {
            return d_idx_tangent_.at(ex.uid());
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        }
    }
};

def_ptr(ocp);
} // namespace moto

extern template void moto::ocp::add<const moto::shared_expr &>(const moto::shared_expr &ex);
extern template void moto::ocp::add<const moto::shared_expr>(const moto::shared_expr &&ex);
extern template void moto::ocp::add<moto::shared_expr>(moto::shared_expr &&ex);

#endif // __MOTO_PROBLEM_HPP__