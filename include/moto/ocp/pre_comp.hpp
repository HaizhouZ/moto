#ifndef MOTO_OCP_PRE_COMP_HPP
#define MOTO_OCP_PRE_COMP_HPP

#include <moto/ocp/impl/custom_func.hpp>

namespace moto {
/////////////////////////////////////////////////////////////////////
/**
 * @brief Generated symbolic precompute with node-local cache outputs.
 */
class generic_pre_compute : public generic_custom_func {
    friend class ocp_base;
    var_list authored_inputs_;
    var_list outputs_;
    var_list runtime_outputs_;

    struct derivative_block {
        size_t output_index;
        var input;
        var cache;
    };
    std::vector<derivative_block> derivative_blocks_;

    struct instance_cache;
    std::shared_ptr<instance_cache> instance_cache_;

    generic_pre_compute(const std::string &name,
                        const std::vector<cs::SX> &outputs);

  protected:
    void substitute(const sym &arg, const sym &rhs) override;
    void load_external_impl(const std::string &path = "gen") override;
    clone_ptr clone() const override { return new generic_pre_compute(*this); }

  public:
    static std::shared_ptr<generic_pre_compute>
    create(const std::string &name, const std::vector<cs::SX> &outputs);
    const var_list &outputs() const { return outputs_; }
    const var_list &authored_inputs() const { return authored_inputs_; }

    /**
     * @brief Instantiate this structural precompute on new input symbols.
     *
     * Target value and derivative caches are allocated automatically. Equal
     * input remaps return the same cached instance.
    */
    expr_handle instantiate(const symbol_remap &input_remap);
    /// Positional convenience overload for replacing every authored input.
    expr_handle instantiate(const var_inarg_list &target_inputs);
};

struct pre_compute : public custom_func {
    using custom_func::custom_func;
    pre_compute() = default;
    generic_pre_compute *operator->() const {
        return static_cast<generic_pre_compute *>(func::operator->());
    }
    static pre_compute create(const std::string &name,
                              const std::vector<cs::SX> &outputs) {
        return generic_pre_compute::create(name, outputs);
    }
    pre_compute instantiate(const generic_func::symbol_remap &input_remap) const {
        return expr_cast<generic_pre_compute>(operator->()->instantiate(input_remap));
    }
    pre_compute instantiate(const var_inarg_list &target_inputs) const {
        return expr_cast<generic_pre_compute>(operator->()->instantiate(target_inputs));
    }
};

/** Return the precompute which owns a public value cache, if any. */
expr_handle precompute_producer(const sym &cache);

/**
 * Extend an argument remap with remapped cache outputs from every directly
 * consumed precompute. Upstream precompute instances are created recursively,
 * while every node remains an independent generated function.
 */
generic_func::symbol_remap remap_precompute_dependencies(
    const generic_func &consumer,
    const generic_func::symbol_remap &argument_remap);

/** Add total first derivatives for any registered precompute caches in out. */
bool apply_precompute_chain_rule(generic_func &consumer, const cs::SX &out);

/** Extend value-cache remaps with their upstream primal and private caches. */
generic_func::symbol_remap expand_precompute_remap(
    const generic_func::symbol_remap &remap);

struct post_compute : public custom_func {
    post_compute() = default; ///< default constructor
    post_compute(const std::string &name) : custom_func(generic_custom_func(name, approx_order::none, 0, __post_comp)) {
    }
};
} // namespace moto

#endif // MOTO_OCP_PRE_COMP_HPP
