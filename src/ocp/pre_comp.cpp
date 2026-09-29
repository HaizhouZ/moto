#include <moto/core/external_function.hpp>
#include <moto/ocp/pre_comp.hpp>
#include <moto/utils/codegen.hpp>

#include <map>
#include <mutex>
#include <unordered_map>

namespace moto {
namespace {
std::vector<cs::SX> vectorize_outputs(const std::vector<cs::SX> &outputs) {
    if (outputs.empty())
        throw std::invalid_argument("symbolic precompute requires at least one output");
    std::vector<cs::SX> result;
    result.reserve(outputs.size());
    for (const cs::SX &output : outputs) {
        if (output.is_empty())
            throw std::invalid_argument("symbolic precompute outputs must not be empty");
        result.push_back(cs::SX::vec(output));
    }
    return result;
}

cs::SX pack_outputs(const std::vector<cs::SX> &outputs) {
    return cs::SX::vertcat(vectorize_outputs(outputs));
}

struct cache_derivative {
    var input;
    var storage;
};

std::mutex provenance_mutex;
std::unordered_map<size_t, std::vector<cache_derivative>> cache_provenance;
std::unordered_map<size_t, expr_handle> cache_producers;

void register_provenance(const var &value, const var &input,
                         const var &storage) {
    std::lock_guard lock(provenance_mutex);
    cache_provenance[value->uid()].push_back({input, storage});
}

std::vector<cache_derivative> provenance(const var &value) {
    std::lock_guard lock(provenance_mutex);
    if (auto found = cache_provenance.find(value->uid());
        found != cache_provenance.end())
        return found->second;
    return {};
}

void register_producer(const std::shared_ptr<generic_pre_compute> &producer) {
    std::lock_guard lock(provenance_mutex);
    for (const var &output : producer->outputs()) {
        auto [entry, inserted] = cache_producers.emplace(output->uid(), producer);
        if (!inserted) {
            const auto &existing = entry->second;
            if (existing && existing->uid() != producer->uid())
                throw std::logic_error(fmt::format(
                    "precompute cache {} already has a producer", output->name()));
            entry->second = producer;
        }
    }
}

const var *mapped_symbol(const generic_func::symbol_remap &remap,
                         const var &source) {
    const auto found = std::ranges::find_if(
        remap, [&](const auto &entry) {
            return entry.first->uid() == source->uid();
        });
    return found == remap.end() ? nullptr : &found->second;
}

expr_handle instantiate_dependency(
    const std::shared_ptr<generic_pre_compute> &source,
    generic_func::symbol_remap &available,
    std::unordered_map<size_t, expr_handle> &instances) {
    if (auto found = instances.find(source->uid()); found != instances.end())
        return found->second;

    for (const expr_handle &dependency : source->dep()) {
        auto upstream = std::dynamic_pointer_cast<generic_pre_compute>(
            static_cast<const std::shared_ptr<expr> &>(dependency));
        if (!upstream)
            continue;
        const expr_handle placed =
            instantiate_dependency(upstream, available, instances);
        const auto placed_precompute = expr_cast<generic_pre_compute>(placed);
        for (size_t i = 0; i < upstream->outputs().size(); ++i)
            available.emplace_back(upstream->outputs()[i],
                                   placed_precompute->outputs()[i]);
    }

    generic_func::symbol_remap input_remap;
    for (const var &input : source->authored_inputs())
        if (const var *target = mapped_symbol(available, input);
            target && (*target)->uid() != input->uid())
            input_remap.emplace_back(input, *target);

    expr_handle placed = input_remap.empty()
        ? source->handle()
        : source->instantiate(input_remap);
    instances.emplace(source->uid(), placed);
    return placed;
}
} // namespace

struct generic_pre_compute::instance_cache {
    std::mutex mutex;
    std::map<remap_key, expr_handle> instances;
};

generic_pre_compute::generic_pre_compute(
    const std::string &name, const std::vector<cs::SX> &outputs)
    : generic_custom_func(name, var_inarg_list{}, pack_outputs(outputs),
                          approx_order::zero, __pre_comp),
      instance_cache_(std::make_shared<instance_cache>()) {
    auto normalized = vectorize_outputs(outputs);
    authored_inputs_ = in_args_;
    gen_.task_->value_outputs = normalized;
    outputs_.reserve(normalized.size());
    for (size_t i = 0; i < normalized.size(); ++i) {
        var output = sym::usr_var(
            fmt::format("{}_cache_{}", name, i),
            static_cast<size_t>(normalized[i].numel()));
        outputs_.push_back(output);
        runtime_outputs_.push_back(output);
        add_argument(output);
        skip_unused_arg_check_.insert(output->uid());
    }

    // Differentiate all outputs for one primal together so CasADi traverses
    // the shared SX graph once per upstream primal.  If an authored output
    // consumes another precompute cache, contract its partial derivative with
    // that cache's stored total derivative instead of expanding the producer.
    size_t row_offset = 0;
    std::vector<size_t> output_offsets{0};
    output_offsets.reserve(normalized.size() + 1);
    for (const cs::SX &output : normalized) {
        row_offset += static_cast<size_t>(output.numel());
        output_offsets.push_back(row_offset);
    }
    const cs::SX stacked = cs::SX::vertcat(normalized);
    const var_list source_inputs = in_args_;
    struct total_derivative {
        var input;
        cs::SX value;
    };
    std::map<size_t, total_derivative> totals;
    const auto accumulate = [&totals](const var &input, const cs::SX &term) {
        auto [entry, inserted] = totals.emplace(
            input->uid(), total_derivative{input, term});
        if (!inserted)
            entry->second.value += term;
    };
    for (const var &input : source_inputs) {
        if (!cs::SX::depends_on(stacked, input))
            continue;
        if (in_field(input->field(), primal_fields))
            accumulate(input, utils::cs_codegen::tangent_jacobian(
                                  stacked, *input));

        const auto upstream = provenance(input);
        if (!upstream.empty()) {
            const cs::SX partial = cs::SX::jacobian(stacked, input);
            for (const cache_derivative &derivative : upstream) {
                // Private upstream derivative caches are runtime inputs of
                // this generated function even though they are not present
                // in the authored value expression.
                add_argument(derivative.storage);
                const cs::SX cache_jacobian = cs::SX::reshape(
                    derivative.storage, input->dim(),
                    derivative.input->tdim());
                accumulate(derivative.input,
                           cs::SX::mtimes(partial, cache_jacobian));
            }
        }
    }
    for (auto &[_, total] : totals) {
        for (size_t i = 0; i < normalized.size(); ++i) {
            const cs::SX derivative = total.value(
                cs::Slice(static_cast<casadi_int>(output_offsets[i]),
                          static_cast<casadi_int>(output_offsets[i + 1])),
                cs::Slice());
            if (derivative.is_zero())
                continue;
            const size_t rows = static_cast<size_t>(normalized[i].numel());
            var cache = sym::usr_var(
                fmt::format("{}_cache_{}_d_{}", name, i,
                            total.input->name()),
                rows * total.input->tdim());
            for (const var &dependency :
                 global_registry::infer_args(derivative))
                add_argument(dependency);
            gen_.task_->value_outputs.push_back(cs::SX::vec(derivative));
            runtime_outputs_.push_back(cache);
            derivative_blocks_.push_back({i, total.input, cache});
            add_argument(cache);
            skip_unused_arg_check_.insert(cache->uid());
            register_provenance(outputs_[i], total.input, cache);
        }
    }
}

std::shared_ptr<generic_pre_compute> generic_pre_compute::create(
    const std::string &name, const std::vector<cs::SX> &outputs) {
    auto result = std::shared_ptr<generic_pre_compute>(
        new generic_pre_compute(name, outputs));
    register_producer(result);
    return result;
}

expr_handle precompute_producer(const sym &cache) {
    std::lock_guard lock(provenance_mutex);
    const auto found = cache_producers.find(cache.uid());
    if (found == cache_producers.end())
        return {};
    return found->second;
}

generic_func::symbol_remap remap_precompute_dependencies(
    const generic_func &consumer,
    const generic_func::symbol_remap &argument_remap) {
    generic_func::symbol_remap result = argument_remap;
    generic_func::symbol_remap available = argument_remap;
    std::unordered_map<size_t, expr_handle> instances;
    for (const expr_handle &dependency : consumer.dep()) {
        auto producer = std::dynamic_pointer_cast<generic_pre_compute>(
            static_cast<const std::shared_ptr<expr> &>(dependency));
        if (!producer)
            continue;
        const auto placed = expr_cast<generic_pre_compute>(
            instantiate_dependency(producer, available, instances));
        for (size_t i = 0; i < producer->outputs().size(); ++i) {
            const var &source = producer->outputs()[i];
            const var &target = placed->outputs()[i];
            available.emplace_back(source, target);
            if (consumer.has_arg(*source) && source->uid() != target->uid())
                result.emplace_back(source, target);
        }
    }
    return result;
}

void generic_pre_compute::substitute(const sym &arg, const sym &rhs) {
    const bool runtime_argument = std::ranges::any_of(
        in_args_, [&](const var &input) { return input->uid() == arg.uid(); });
    if (runtime_argument)
        generic_custom_func::substitute(arg, rhs);
    bool metadata_argument = false;
    for (var &input : authored_inputs_)
        if (input->uid() == arg.uid()) {
            input = var(rhs);
            metadata_argument = true;
        }
    for (var &output : outputs_)
        if (output->uid() == arg.uid())
            output = var(rhs);
    for (var &output : runtime_outputs_)
        if (output->uid() == arg.uid())
            output = var(rhs);
    for (derivative_block &block : derivative_blocks_) {
        if (block.input->uid() == arg.uid()) {
            block.input = var(rhs);
            metadata_argument = true;
        }
        if (block.cache->uid() == arg.uid())
            block.cache = var(rhs);
    }
    if (!runtime_argument && !metadata_argument)
        throw std::runtime_error(fmt::format(
            "precompute {} substitute failed: argument {} not found",
            name_, arg.name()));
}

expr_handle generic_pre_compute::instantiate(const symbol_remap &input_remap) {
    for (const auto &[from, _] : input_remap) {
        if (std::ranges::find(runtime_outputs_, from) != runtime_outputs_.end())
            throw std::invalid_argument(fmt::format(
                "precompute {} instantiate accepts input mappings only; "
                "output cache {} is managed by Moto", name_, from->name()));
        if (std::ranges::find(authored_inputs_, from) == authored_inputs_.end())
            throw std::invalid_argument(fmt::format(
                "precompute {} instantiate source {} is not an input",
                name_, from->name()));
    }
    const auto normalized = normalize_argument_remap(input_remap);
    if (normalized.empty())
        return handle();

    std::lock_guard lock(instance_cache_->mutex);
    if (auto found = instance_cache_->instances.find(normalized.key);
        found != instance_cache_->instances.end())
        return found->second;

    symbol_remap full_remap = input_remap;
    var_list target_runtime_outputs;
    target_runtime_outputs.reserve(runtime_outputs_.size());
    for (size_t i = 0; i < runtime_outputs_.size(); ++i) {
        const var &source = runtime_outputs_[i];
        var target = sym::usr_var(
            fmt::format("{}_instance_{}_cache_{}", name_, uid(), i),
            source->dim());
        target_runtime_outputs.push_back(target);
        full_remap.emplace_back(source, target);
    }

    const auto remapped_input = [&](const var &source) -> var {
        if (auto found = normalized.entries.find(source->uid());
            found != normalized.entries.end())
            return found->second.second;
        return source;
    };
    for (const derivative_block &block : derivative_blocks_) {
        const auto source_cache = std::ranges::find(runtime_outputs_, block.cache);
        if (source_cache == runtime_outputs_.end())
            throw std::logic_error("precompute derivative cache is not a runtime output");
        register_provenance(
            target_runtime_outputs.at(block.output_index),
            remapped_input(block.input),
            target_runtime_outputs.at(
                static_cast<size_t>(source_cache - runtime_outputs_.begin())));
    }

    expr_handle instance = generic_func::reuse_remap(full_remap);
    register_producer(expr_cast<generic_pre_compute>(instance));
    instance_cache_->instances.emplace(normalized.key, instance);
    return instance;
}

expr_handle generic_pre_compute::instantiate(
    const var_inarg_list &target_inputs) {
    if (authored_inputs_.size() != target_inputs.size())
        throw std::invalid_argument(fmt::format(
            "precompute {} instantiate expected {} inputs, got {}",
            name_, authored_inputs_.size(), target_inputs.size()));
    symbol_remap remap;
    remap.reserve(authored_inputs_.size());
    for (size_t i = 0; i < authored_inputs_.size(); ++i)
        remap.emplace_back(authored_inputs_[i], var(target_inputs[i].get()));
    return instantiate(remap);
}

bool apply_precompute_chain_rule(generic_func &consumer, const cs::SX &out) {
    struct total_derivative {
        var input;
        cs::SX value;
    };
    std::map<size_t, total_derivative> totals;
    bool used = false;
    for (const var &cache : global_registry::infer_args(out)) {
        for (const cache_derivative &derivative : provenance(cache)) {
            used = true;
            const cs::SX partial = cs::SX::jacobian(out, cache);
            const cs::SX cache_jacobian = cs::SX::reshape(
                derivative.storage, cache->dim(), derivative.input->tdim());
            const cs::SX term = cs::SX::mtimes(partial, cache_jacobian);
            auto [entry, inserted] = totals.emplace(
                derivative.input->uid(),
                total_derivative{derivative.input, term});
            if (!inserted)
                entry->second.value += term;
        }
    }
    for (auto &[_, total] : totals) {
        if (cs::SX::depends_on(out, total.input))
            total.value += utils::cs_codegen::tangent_jacobian(
                out, *total.input);
        consumer.set_analytic_jacobian(*total.input, total.value);
    }
    return used;
}

generic_func::symbol_remap expand_precompute_remap(
    const generic_func::symbol_remap &remap) {
    generic_func::symbol_remap expanded = remap;
    for (const auto &[from, to] : remap) {
        const auto source = provenance(from);
        const auto target = provenance(to);
        if (source.empty() || target.empty())
            continue;
        if (source.size() != target.size())
            throw std::invalid_argument(fmt::format(
                "cannot remap precompute cache {} to {}: derivative layouts "
                "have {} and {} blocks", from->name(), to->name(),
                source.size(), target.size()));
        for (size_t i = 0; i < source.size(); ++i) {
            if (source[i].input->dim() != target[i].input->dim() ||
                source[i].input->tdim() != target[i].input->tdim() ||
                source[i].storage->dim() != target[i].storage->dim())
                throw std::invalid_argument(fmt::format(
                    "cannot remap precompute cache {} to {}: derivative block "
                    "{} is structurally incompatible", from->name(),
                    to->name(), i));
            expanded.emplace_back(source[i].input, target[i].input);
            expanded.emplace_back(source[i].storage, target[i].storage);
        }
    }
    return expanded;
}

void generic_pre_compute::load_external_impl(const std::string &path) {
    const std::string func_name =
        (gen_.task_ && !gen_.task_->func_name.empty())
            ? gen_.task_->func_name
            : name_;
    const std::string artifact_path =
        gen_.task_ && gen_.task_->eval_artifact_dir &&
                !gen_.task_->eval_artifact_dir->empty()
            ? *gen_.task_->eval_artifact_dir
            : path;
    ext_func eval(func_name, artifact_path);
    std::vector<size_t> output_indices;
    output_indices.reserve(runtime_outputs_.size());
    for (const var &output : runtime_outputs_) {
        const auto found = std::ranges::find(in_args_, output);
        if (found == in_args_.end())
            throw std::logic_error(fmt::format(
                "precompute {} output {} is not a runtime argument",
                name_, output->name()));
        output_indices.push_back(
            static_cast<size_t>(found - in_args_.begin()));
    }
    custom_call = [eval = std::move(eval),
                   output_indices = std::move(output_indices)](
                      func_arg_map &data) mutable {
        if (output_indices.size() == 1) {
            eval.invoke(data.in_arg_data(), data[output_indices.front()]);
            return;
        }
        std::vector<vector_ref> outputs;
        outputs.reserve(output_indices.size());
        for (size_t index : output_indices)
            outputs.emplace_back(data[index]);
        eval.invoke(data.in_arg_data(), outputs);
    };
}
} // namespace moto
