#ifndef __MOTO_PROBLEM_HPP__
#define __MOTO_PROBLEM_HPP__

#include <array>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <moto/core/expr.hpp>

namespace moto {
class ocp;
def_ptr(ocp);
/**
 * @brief problem formulation of an OCP stage
 *
 */
class ocp {
  protected:
    ocp() { uid_.set_inc(); };     // default constructor
    ocp(const ocp &rhs) = default; // copy constructor
    bool add_impl(expr &);
    /// @brief maintain the order of primal variables by dynamics, 
    /// required by block-matrix coputation
    void maintain_order();
    static size_t max_uid; ///< uid used to index global expressions
    bool finalized_ = false;
    utils::unique_id<ocp> uid_;
    /// collection of all expressions in the problem
    std::array<expr_list, field::num> expr_, disabled_expr_, pruned_expr_;
    /// data index of expr in serialized vector, by uid
    std::unordered_map<size_t, size_t> flatten_idx_;
    /// data index of tagent space var of expr in serialized vector, by uid
    std::unordered_map<size_t, size_t> flatten_tidx_;
    /// position of expr in expr_ e.g., no. xxx, by uid
    std::unordered_map<size_t, size_t> pos_by_uid_;
    /// dimension of each field
    std::array<size_t, field::num> dim_{};
    /// tangent space dimension of each field
    std::array<size_t, field::num_prim> tdim_{};
    /// set of uids to check for duplicates when adding expr
    std::unordered_set<size_t> uids_, disabled_uids_, pruned_uids_;

    void set_dim_and_idx();
    void finalize();
    inline void field_access_guard() const {
        assert(finalized_ && "Cannot access before the problem is finalized. "
                             "Please call finalize() before accessing expressions.");
    }
    /// @brief get data pointer for expr in the serialized vector
    scalar_t *get_data_ptr(scalar_t *data, const expr &ex) const {
        return data + get_expr_start(ex);
    }
    /// @deprecated, use get_data_ptr instead
    // scalar_t *get_data_ptr(scalar_t *data, const expr &ex, size_t offset) const {
    //     return data + get_expr_start(ex) * offset;
    // }

  public:
    CONST_PROPERTY(uid); ///< getter for uid
    /// @brief getter for enabled expressions in field f
    const auto &exprs(size_t f) const { return expr_.at(f); }
    /// @brief getter for position of expr in its field, by uid
    const auto &pos(const expr &ex) const {
        field_access_guard();
        return pos_by_uid_.at(ex.uid());
    }
    /// @brief getter for dimension of field f
    size_t dim(size_t f) const {
        field_access_guard();
        return dim_.at(f);
    }
    /// @brief getter for num of exprs in field f
    size_t num(size_t f) const { return expr_[f].size(); }
    /// @brief getter for tangent space dimension of field f
    size_t tdim(size_t f) const {
        field_access_guard();
        return tdim_.at(f);
    }
    /// @brief check if expr is in the problem
    bool contains(const expr &ex, bool include_sub_prob = true) const;
    /// @brief check if expr is active in the problem
    bool is_active(const expr &ex, bool include_sub_prob = true) const;
    /// @brief wait until all expressions in the problem are ready, throw if any expression fails to be ready
    void wait_until_ready();
    /// @brief print a summary of the problem
    void print_summary();

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
    void add(T &&ex) { add_impl(ex); }
    /// add multiple exprs
    void add(const expr_inarg_list &exprs) {
        for (expr &ex : exprs) {
            add(ex);
        }
    }

    /// add a sub-problem
    void add(const ocp_ptr_t &sub) {
        sub_probs_.push_back(sub);
    }

    /**
     * @brief get start index of expr in its field
     */
    size_t get_expr_start(const expr &ex) const {
        try {
            field_access_guard();
            return flatten_idx_.at(ex.uid());
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        }
    }

    size_t get_expr_start_tangent(const expr &ex) const {
        try {
            field_access_guard();
            return flatten_tidx_.at(ex.uid());
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        }
    }

    static auto create() { return std::shared_ptr<ocp>(new ocp()); }

    /// @brief configuration for cloning an ocp
    /// @note deactivate_list will treated as forced deactivation, thus automatic-pruning will not re-enable them
    /// @note activate_list will re-enable pruned expressions if their args are active
    ///      but will NOT re-enable previously user-disabled expressions
    /// @warning if an expr shows up in both lists, it will be deactivated
    struct active_status_config {
        expr_inarg_list deactivate_list; ///< list of expressions to be deactivated in the cloned problem
        expr_inarg_list activate_list;   ///< list of expressions to be re-activated in the cloned problem
        bool empty() const { return deactivate_list.empty() && activate_list.empty(); }
    };

    ocp_ptr_t clone(const active_status_config &config = {}) const;
    void update_active_status(const active_status_config &config, bool update_sub_probs = true);

  private:
    bool allow_inconsistent_dynamics_ = false; ///< allow inconsistent dynamics when updating active status
    bool automatic_reorder_primal_ = true;     ///< automatically reorder primal variables by dynamics
    std::vector<ocp_ptr_t> sub_probs_;         ///< list of sub-problems owned by this problem
  public:
    PROPERTY(allow_inconsistent_dynamics) ///< getter and setter for allow_inconsistent_dynamics
    PROPERTY(automatic_reorder_primal)    ///< getter and setter for automatic_reorder_primal
    const auto &sub_probs() const { return sub_probs_; }
};

} // namespace moto

extern template void moto::ocp::add<const moto::shared_expr &>(const moto::shared_expr &ex);
extern template void moto::ocp::add<const moto::shared_expr>(const moto::shared_expr &&ex);
extern template void moto::ocp::add<moto::shared_expr>(moto::shared_expr &&ex);

#endif // __MOTO_PROBLEM_HPP__