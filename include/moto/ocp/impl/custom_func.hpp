#ifndef MOTO_OCP_IMPL_CUSTOM_FUNC_HPP
#define MOTO_OCP_IMPL_CUSTOM_FUNC_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto {
namespace impl {

class custom_func : public func {
  public:
    /**
     * @brief constructor for non-approximation functions
     *
     * @param name name of the function
     * @param field field, must explicitly belong to the non-approximation fields
     */
    custom_func(const std::string &name, field_t field)
        : custom_func(name, approx_order::none, 0, field) {
    }
    /**
     * @brief constructor for non-approximation functions with dimension
     *
     * @param name name of the function
     * @param order order of the approximation, default is none
     * @param dim dimension of the function, default is 0
     * @param field field, must explicitly belong to the non-approximation fields
     */
    custom_func(const std::string &name, approx_order order, size_t dim, field_t field)
        : func(name, order, dim, field) {
        if (in_field(field, moto::func_fields) || field == __undefined) {
            throw std::runtime_error(fmt::format("func {} field type {} not qualified as custom function",
                                                 name_, field::name(field)));
        }
        // make a default create_custom_data
        create_custom_data = [this](sym_data &primal, shared_data &shared) {
            return create_arg_map(primal, shared);
        };
    }

    using func::func;
    /**
     * @brief create the argument mapping for the custom function
     * 
     * @param primal sym_data symbolic variable data
     * @param shared shared_data the shared data of the problem
     * @return func_arg_map_ptr_t 
     */
    func_arg_map_ptr_t create_arg_map(sym_data &primal, shared_data &shared) {
        // create the func_arg_map for the custom function
        return std::make_unique<func_arg_map>(primal, shared, *this);
    }

    /// @brief callback to make data（for non-approx) @note will not be called in @ref create_approx_map
    std::function<func_arg_map_ptr_t(sym_data &, shared_data &)> create_custom_data;
    /// @brief callback to call a non-approximation function
    std::function<void(func_arg_map &)> custom_call;
};

} // namespace impl
} // namespace moto

#endif // MOTO_OCP_IMPL_CUSTOM_FUNC_HPP