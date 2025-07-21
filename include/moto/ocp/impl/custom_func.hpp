#ifndef MOTO_OCP_IMPL_CUSTOM_FUNC_HPP
#define MOTO_OCP_IMPL_CUSTOM_FUNC_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto {

class custom_func : public func {
  protected:
    void finalize_impl() override {
        if (in_field(field_, moto::func_fields) || field_ == __undefined) {
            throw std::runtime_error(fmt::format("func {} field type {} not qualified as custom function - finalization failed",
                                                 name_, field::name(field_)));
        }
    }

  public:
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

template <typename derived>
using custom_func_derived = func_derived<derived, custom_func>;

} // namespace moto

#endif // MOTO_OCP_IMPL_CUSTOM_FUNC_HPP