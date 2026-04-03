#ifndef MOTO_OCP_IMPL_CUSTOM_FUNC_HPP
#define MOTO_OCP_IMPL_CUSTOM_FUNC_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto {
class generic_custom_func;
struct usr_func;
struct pre_compute;
struct custom_func : public func {
    using func::func; ///< inherit base constructor
    generic_custom_func *operator->() const;
};

class generic_custom_func : public generic_func {

  protected:
    void finalize_impl() override {
        if (in_field(field_, moto::func_fields) || field_ == __undefined) {
            throw std::runtime_error(fmt::format("func {} field type {} not qualified as custom function - finalization failed",
                                                 name_, field::name(field_)));
        }
        generic_func::finalize_impl();
    }
    using wrapper_type = custom_func;

    friend struct custom_func; ///< allow custom_func to access private members

  public:
    using generic_func::generic_func;
    /**
     * @brief create the argument mapping for the custom function
     *
     * @param primal sym_data symbolic variable data
     * @param shared shared_data the shared data of the problem
     * @return func_arg_map_ptr_t
     */
    virtual func_arg_map_ptr_t create_custom_data(sym_data &primal, lag_data &raw, shared_data &shared) const {
        return std::make_unique<func_arg_map>(primal, shared, *this);
    }
    /// @brief callback to make data（for non-approx) @note will not be called in @ref create_approx_data
    /// @brief callback to call a non-approximation function
    std::function<void(func_arg_map &)> custom_call;
};
inline generic_custom_func *custom_func::operator->() const {
    return static_cast<generic_custom_func *>(func::operator->());
} ///< convert to generic_custom_func
} // namespace moto

#endif // MOTO_OCP_IMPL_CUSTOM_FUNC_HPP
