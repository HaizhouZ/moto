#ifndef MOTO_OCP_IMPL_CUSTOM_FUNC_HPP
#define MOTO_OCP_IMPL_CUSTOM_FUNC_HPP

#include <moto/ocp/impl/func.hpp>

namespace moto {

class custom_func : public func {

  public:
    struct impl : public func::impl {
        /// @brief callback to make data（for non-approx) @note will not be called in @ref create_approx_map
        std::function<func_arg_map_ptr_t(sym_data &, shared_data &)> create_custom_data_;
        /// @brief callback to call a non-approximation function
        std::function<void(func_arg_map &)> custom_call_;

        impl() = default;           ///< default constructor
        impl(impl &&rhs) = default; ///< move constructor
        explicit impl(func::impl &&rhs)
            : func::impl(std::move(rhs)) {}

        void finalize_impl() override {
            if (in_field(field_, moto::func_fields) || field_ == __undefined) {
                throw std::runtime_error(fmt::format("func {} field type {} not qualified as custom function - finalization failed",
                                                     name_, field::name(field_)));
            }
        }
    };

    using func::func;
    /**
     * @brief create the argument mapping for the custom function
     *
     * @param primal sym_data symbolic variable data
     * @param shared shared_data the shared data of the problem
     * @return func_arg_map_ptr_t
     */
    func_arg_map_ptr_t create_arg_map(sym_data &primal, shared_data &shared) const {
        // create the func_arg_map for the custom function
        return std::make_unique<func_arg_map>(primal, shared, *this);
    }

    IMPL_ATTR_GETTER(create_custom_data, custom_func);
    IMPL_ATTR_GETTER(custom_call, custom_func);
};

} // namespace moto

#endif // MOTO_OCP_IMPL_CUSTOM_FUNC_HPP