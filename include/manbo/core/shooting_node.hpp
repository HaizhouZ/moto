#ifndef __NODE__
#define __NODE__

#include <vector>
#include <memory>
#include <map>

#include <manbo/core/expression_base.hpp>

namespace manbo {
/**
 * @brief enum class of field types
 *
 */
enum class fieldType : size_t {
    dyn = 0,  // dynamic model
    cost,     // "running cost"
    cost_e,   // "terminal cost"
    constr,   // "constraints"
    constr_e  // "terminal constraints"
};
/**
 * @brief problem formulation of an OCP stage
 *
 */
class problemFormulation {
   private:
    typedef std::map<std::string, std::shared_ptr<exprBase>> expr_collection_t;
    typedef std::map<std::string, std::pair<size_t, size_t>> expr_idx_t;
    std::vector<expr_collection_t> expr_;
    std::vector<expr_idx_t> idx_;

   public:
    problemFormulation() {
        expr_.resize(magic_enum::enum_count<fieldType>());
        idx_.resize(magic_enum::enum_count<fieldType>());
    }

    /**
     * @brief add expr to problem formulation
     *
     * @param expr expression to be added
     * @param field in [fieldType]
     */
    void add_expr(std::shared_ptr<exprBase> expr, fieldType field) {
        size_t _field = size_t(field);
        const auto& _name = expr->name();
        auto n0 = expr_[_field].size();
        expr_[_field][_name] = expr;
        auto n1 = expr_[_field].size();
        idx_[_field][_name] = std::make_pair(n0, n1);
    }

    /**
     * @brief get the idx of an expr named by [name]
     *
     * @param name name of the expression
     * @param field type of the expression
     * @return std::pair<size_t, size_t> [start, end) of the expression
     */
    std::pair<size_t, size_t> get_idx(const std::string& name, fieldType field) {
        try {
            return idx_[size_t(field)][name];
        } catch (const std::exception& e) {
            throw std::runtime_error(fmt::format(
                "Cannot get idx of {0} in field {1}", name, magic_enum::enum_name(field)));
        }
    }
    /**
     * @brief get the idx of an expr named by [name]
     *
     * @param expr expression to look up
     * @param field type of the expression
     * @return std::pair<size_t, size_t> [start, end) of the expression
     */
    std::pair<size_t, size_t> get_idx(std::shared_ptr<exprBase> expr, fieldType field) {
        return get_idx(expr->name(), field);
    }
};
/**
 * @brief shooting node in an OCP
 * @todo data collection/serialization/deserialization should be finished in this node!
 */
class shootingNode {
   public:
    shootingNode(std::shared_ptr<problemFormulation> formulation)
        : formulation_(formulation) {
    }
    void createData();

   private:
    std::vector<std::shared_ptr<shootingNode>> next_;
    std::shared_ptr<problemFormulation> formulation_;
};
}  // namespace manbo

#endif /*__NODE_*/