#include <algorithm>
#include <array>
#include <map>
#include <moto/ocp/dynamics.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>

namespace moto {
INIT_UID_(ocp_base);

namespace {
constexpr auto status_func_fields = concat_fields(func_fields, custom_func_fields);

bool has_active_primal_arg(const generic_func &func, const ocp_base *prob) {
    for (const sym &arg : func.in_args()) {
        if (in_field(arg.field(), primal_fields) && prob->is_active(arg)) {
            return true;
        }
    }
    return false;
}

bool can_be_active_by_status(const generic_func &func, ocp_base *prob) {
    return has_active_primal_arg(func, prob) && func.check_enable(prob);
}

const generic_func *as_generic_func(const expr_handle &ex) {
    return ex ? dynamic_cast<const generic_func *>(ex.get()) : nullptr;
}

bool is_pure_y_func(const generic_func &func) {
    bool has_y = false;
    for (const sym &arg : func.in_args()) {
        if (!in_field(arg.field(), primal_fields)) {
            continue;
        }
        if (arg.field() != __y) {
            return false;
        }
        has_y = true;
    }
    return has_y;
}

constexpr unsigned role_bit(stage_expr_role role) {
    return 1u << static_cast<unsigned>(role);
}

casadi::Sparsity sparsity_from_layout(size_t rows, size_t cols,
                                     const sparse_layout_plan &layout) {
    std::vector<casadi_int> rr, cc;
    for (const auto &panel : layout.panels) {
        if (panel.pattern == sparsity::dense) {
            for (size_t c = panel.col; c < panel.col + panel.cols; ++c)
                for (size_t r = panel.row; r < panel.row + panel.rows; ++r) {
                    rr.push_back(static_cast<casadi_int>(r));
                    cc.push_back(static_cast<casadi_int>(c));
                }
        } else if (panel.pattern == sparsity::diag ||
                   panel.pattern == sparsity::eye) {
            for (size_t i = 0; i < std::min(panel.rows, panel.cols); ++i) {
                rr.push_back(static_cast<casadi_int>(panel.row + i));
                cc.push_back(static_cast<casadi_int>(panel.col + i));
            }
        }
    }
    return casadi::Sparsity::triplet(static_cast<casadi_int>(rows),
                                     static_cast<casadi_int>(cols), rr, cc);
}

casadi::Sparsity sparsity_from_layout(
    const linear_backend::matrix_layout &layout) {
    std::vector<casadi_int> rr, cc;
    for (const auto &panel : layout.panels) {
        if (panel.pattern == sparsity::dense) {
            for (size_t c = 0; c < panel.cols; ++c)
                for (size_t r = 0; r < panel.rows; ++r) {
                    rr.push_back(static_cast<casadi_int>(panel.row_offset + r));
                    cc.push_back(static_cast<casadi_int>(panel.col_offset + c));
                }
        } else if (panel.pattern == sparsity::diag ||
                   panel.pattern == sparsity::eye) {
            for (size_t i = 0; i < std::min(panel.rows, panel.cols); ++i) {
                rr.push_back(static_cast<casadi_int>(panel.row_offset + i));
                cc.push_back(static_cast<casadi_int>(panel.col_offset + i));
            }
        }
    }
    return casadi::Sparsity::triplet(
        static_cast<casadi_int>(layout.rows),
        static_cast<casadi_int>(layout.cols), rr, cc);
}

cs::MX symbolic_block(std::string name, size_t rows, size_t cols,
                      const sparse_layout_plan &layout, field_t equation,
                      field_t variable, lifted_graph_program &program) {
    cs::MX input = cs::MX::sym(
        std::move(name), sparsity_from_layout(rows, cols, layout));
    linear_backend::matrix_layout runtime_layout{rows, cols, {}};
    for (const auto pattern :
         std::array{sparsity::dense, sparsity::diag, sparsity::eye})
        for (const auto &panel : layout.panels)
            if (panel.pattern == pattern)
                runtime_layout.panels.push_back({
                    panel.pattern, panel.row, panel.col,
                    panel.rows, panel.cols});
    program.inputs.push_back(input);
    program.input_layouts.push_back(std::move(runtime_layout));
    program.input_bindings.push_back({
        lifted_graph_input_binding::kind::jacobian_panel,
        equation, variable, 0, {}});
    return input;
}

std::vector<sparse_block_spec> blocks_from_sparsity(
    const casadi::Sparsity &sp) {
    const size_t rows = static_cast<size_t>(sp.size1());
    const size_t cols = static_cast<size_t>(sp.size2());
    const size_t nnz = static_cast<size_t>(sp.nnz());
    if (!nnz) return {};
    if (nnz == rows * cols)
        return {{0, 0, rows, cols, sparsity::dense}};

    std::vector<casadi_int> triplet_rows, triplet_cols;
    sp.get_triplet(triplet_rows, triplet_cols);
    std::vector<std::vector<size_t>> by_row(rows);
    for (size_t i = 0; i < triplet_rows.size(); ++i)
        by_row.at(static_cast<size_t>(triplet_rows[i])).push_back(
            static_cast<size_t>(triplet_cols[i]));
    for (auto &entries : by_row)
        std::ranges::sort(entries);

    const bool singleton_rows = std::ranges::all_of(
        by_row, [](const auto &entries) { return entries.size() <= 1; });
    std::vector<sparse_block_spec> result;
    if (singleton_rows) {
        size_t r = 0;
        while (r < rows) {
            if (by_row[r].empty()) {
                ++r;
                continue;
            }
            const size_t start_row = r;
            const size_t start_col = by_row[r][0];
            while (r + 1 < rows && by_row[r + 1].size() == 1 &&
                   by_row[r + 1][0] == start_col + (r + 1 - start_row))
                ++r;
            const size_t length = r - start_row + 1;
            result.push_back({start_row, start_col, length, length,
                              sparsity::diag});
            ++r;
        }
        return result;
    }

    struct active_rectangle {
        size_t index;
        size_t last_row;
    };
    std::map<std::pair<size_t, size_t>, active_rectangle> active;
    for (size_t r = 0; r < rows; ++r) {
        std::map<std::pair<size_t, size_t>, active_rectangle> next;
        const auto &entries = by_row[r];
        for (size_t i = 0; i < entries.size();) {
            const size_t begin = entries[i];
            size_t end = begin + 1;
            while (++i < entries.size() && entries[i] == end) ++end;
            const auto key = std::pair{begin, end};
            if (auto it = active.find(key);
                it != active.end() && it->second.last_row + 1 == r) {
                auto &block = result[it->second.index];
                ++block.rows;
                next.emplace(key, active_rectangle{it->second.index, r});
            } else {
                result.push_back(
                    {r, begin, 1, end - begin, sparsity::dense});
                next.emplace(key,
                             active_rectangle{result.size() - 1, r});
            }
        }
        active = std::move(next);
    }
    return result;
}

sparse_layout_plan layout_from_sparsity(const casadi::Sparsity &sp) {
    auto blocks = blocks_from_sparsity(sp);
    return make_sparse_layout_plan(blocks);
}

void append_graph_outputs(const cs::MX &value,
                          const sparse_layout_plan &layout,
                          std::vector<cs::MX> &outputs) {
    for (const auto pattern : std::array{sparsity::dense, sparsity::diag})
        for (const auto &panel : layout.panels) {
            if (panel.pattern != pattern) continue;
            const auto block = value(
                cs::Slice(static_cast<casadi_int>(panel.row),
                          static_cast<casadi_int>(panel.row + panel.rows)),
                cs::Slice(static_cast<casadi_int>(panel.col),
                          static_cast<casadi_int>(panel.col + panel.cols)));
            outputs.push_back(cs::MX::densify(
                pattern == sparsity::dense ? block : cs::MX::diag(block)));
        }
}

} // namespace

ocp_base::ocp_base() { uid_.set_inc(); }

ocp_base::ocp_base(const ocp_base &rhs)
    : field_layout_store<expr_list>(rhs),
      finalized_(rhs.finalized_),
      uid_(rhs.uid_),
      disabled_expr_(rhs.disabled_expr_),
      pruned_expr_(rhs.pruned_expr_),
      linear_profile_(rhs.linear_profile_),
      uids_(rhs.uids_),
      disabled_uids_(rhs.disabled_uids_),
      pruned_uids_(rhs.pruned_uids_),
      allow_inconsistent_dynamics_(rhs.allow_inconsistent_dynamics_),
      automatic_reorder_primal_(rhs.automatic_reorder_primal_) {}

ocp_base::~ocp_base() = default;

bool ocp_base::add_impl(expr_handle ex) {
    if (!ex) {
        throw std::runtime_error("Cannot add null expression to problem");
    }
    std::string reason;
    if (!accepts_term(ex, &reason)) {
        if (reason.empty()) {
            reason = "expression is incompatible with this problem type";
        }
        throw std::runtime_error(fmt::format(
            "Cannot add expression {} uid {} to problem uid {}: {}",
            ex->name(), ex->uid(), uid_, reason));
    }
    size_t _uid = ex->uid();
    if (!contains(*ex)) {
        if (!ex->finalize()) {
            throw std::runtime_error(fmt::format("cannot finalize expr {} uid {}", ex->name(), ex->uid()));
        }
        const auto &dep = ex->dep();
        if (!dep.empty()) {
            if (ex->field() == __dyn && !allow_inconsistent_dynamics_) {
                const auto *candidate = dynamic_cast<const generic_dynamics *>(ex.get());
                if (candidate) for (const expr_handle &existing : field_entries(__dyn)) {
                    const auto *dyn = dynamic_cast<const generic_dynamics *>(existing.get());
                    if (!dyn) continue;
                    for (const expr_handle &dependency : dep) {
                        const auto *arg =
                            dynamic_cast<const sym *>(dependency.get());
                        if (!arg || !dyn->has_arg(*arg)) continue;
                        if (arg->field() == __x || arg->field() == __y) {
                            throw std::runtime_error(fmt::format(
                                "Dynamics {} and {} overlap state {} uid {} in {}",
                                candidate->name(), dyn->name(), arg->name(),
                                arg->uid(), arg->field()));
                        }
                        if (arg->field() == __u &&
                            (!candidate->input_shared(*arg) ||
                             !dyn->input_shared(*arg))) {
                            throw std::runtime_error(fmt::format(
                                "Dynamics {} and {} share input {} uid {} without "
                                "mark_shared_inputs() on both dynamics",
                                candidate->name(), dyn->name(), arg->name(),
                                arg->uid()));
                        }
                    }
                }
            }
            for (expr &arg : dep) {
                if (!contains(arg)) {
                    add_impl(arg);
                }
            }
        }
        finalized_ = false;
        if (ex->default_active_status()) {
            uids_.insert(_uid);
            append_entry(std::move(ex));
        } else {
            disabled_uids_.insert(_uid);
            disabled_expr_[ex->field()].emplace_back(std::move(ex));
        }
        on_modified();
        return true;
    }
    return false;
}
bool ocp_base::contains(const expr &ex) const {
    return uids_.contains(ex.uid()) ||
           disabled_uids_.contains(ex.uid()) ||
           pruned_uids_.contains(ex.uid());
}
bool ocp_base::is_active(const expr &ex) const { return uids_.contains(ex.uid()); }
const expr_list &ocp_base::exprs(size_t f) const { return field_entries(f); }
size_t ocp_base::pos(const expr &ex) const {
    field_read_guard();
    return entry_index(ex);
}
size_t ocp_base::dim(size_t f) const { field_read_guard(); return field_dim(f); }
size_t ocp_base::num(size_t f) const { return field_entry_count(f); }
size_t ocp_base::tdim(size_t f) const { field_read_guard(); return field_tdim(f); }
size_t ocp_base::get_expr_start(const expr &ex) const {
    field_read_guard();
    return field_start(ex);
}
size_t ocp_base::get_expr_start_tangent(const expr &ex) const {
    field_read_guard();
    return field_tangent_start(ex);
}
void ocp_base::finalize() {
    static std::mutex finalize_mutex_;
    std::lock_guard lock(finalize_mutex_);
    if (!finalized_) {
        if (!field_empty(__dyn) && automatic_reorder_primal_)
            maintain_order();
        rebuild_layout();
        for (const expr_handle &entry : exprs(__dyn)) {
            const auto *dyn = dynamic_cast<const generic_dynamics *>(entry.get());
            if (!dyn) continue;
            size_t nx = 0, ny = 0;
            for (const sym &arg : dyn->in_args()) {
                if (!is_active(arg)) continue;
                if (arg.field() == __x) nx += arg.tdim();
                if (arg.field() == __y) ny += arg.tdim();
            }
            if (!nx || !ny || nx != ny || dyn->dim() != ny) {
                throw std::runtime_error(fmt::format(
                    "Dynamics {} requires equal nonzero tangent dimensions: "
                    "x={}, y={}, residual={}",
                    dyn->name(), nx, ny, dyn->dim()));
            }
        }
        this->finalized_ = true;
        build_linear_profile();
    }
}

void ocp_base::build_linear_profile() {
    linear_profile_ = {};
    std::unordered_map<size_t, std::vector<sparse_block_spec>> blocks;
    struct jacobian_source {
        const generic_func *function = nullptr;
        size_t argument = 0;
        sp_info local;
    };
    std::unordered_map<size_t, std::vector<jacobian_source>> jacobian_sources;
    const auto add = [&blocks](linear_target target, field_t a, field_t b,
                               sparse_block_spec block) {
        blocks[ocp_linear_profile::key(target, a, b)].push_back(block);
    };
    const auto add_jacobian = [&](const generic_func &function,
                                  size_t argument, const sp_info &local) {
        const sym &variable = function.in_args(argument);
        const size_t key = ocp_linear_profile::key(
            linear_target::jacobian, function.field(), variable.field());
        blocks[key].push_back({
            get_expr_start(function) + local.row_offset,
            get_expr_start_tangent(variable) + local.col_offset,
            local.rows, local.cols, local.pattern});
        jacobian_sources[key].push_back({&function, argument, local});
    };
    for (const auto f : primal_fields) {
        if (!tdim(f))
            continue;
        const sparse_block_spec diagonal{0, 0, tdim(f), tdim(f), sparsity::diag};
        add(linear_target::lag_hessian, f, f, diagonal);
        add(linear_target::hessian_modification, f, f, diagonal);
    }
    for (const auto ff : func_fields) {
        for (const generic_func &f : exprs(ff)) {
            const auto &args = f.in_args();
            if (f.order() >= approx_order::first && ff != __cost) {
                const auto *structured =
                    ff == __dyn ? dynamic_cast<const generic_lifted *>(&f)
                                : nullptr;
                const bool has_structured_panels =
                    structured &&
                    !structured->jacobian_panel_sparsity().empty();
                if (has_structured_panels) {
                    for (const auto &[i, sp] :
                         structured->jacobian_panel_sparsity()) {
                        const auto &arg = args[i];
                        if (arg->field() >= field::num_prim || !is_active(arg))
                            continue;
                        add_jacobian(f, i, sp);
                    }
                } else {
                    for (size_t i : range(args.size())) {
                        const auto &arg = args[i];
                        if (arg->field() >= field::num_prim || !is_active(arg))
                            continue;
                        const auto &sp = f.jac_sparsity()[i];
                        if (sp.pattern == sparsity::unknown)
                            continue;
                        add_jacobian(f, i, sp);
                    }
                }
            }
            if (f.order() < approx_order::second && !in_field(ff, ineq_soft_constr_fields))
                continue;
            const auto target = ff == __cost ? linear_target::lag_hessian
                                             : linear_target::hessian_modification;
            if (!f.hess_panel_sparsity().empty()) {
                for (const auto &[i, j, sp] : f.hess_panel_sparsity()) {
                    const auto fi = args[i]->field(), fj = args[j]->field();
                    if (fi >= field::num_prim || fj >= field::num_prim || fi < fj ||
                        !is_active(args[i]) || !is_active(args[j])) continue;
                    add(target, fi, fj,
                        {get_expr_start_tangent(args[i]) + sp.row_offset,
                         get_expr_start_tangent(args[j]) + sp.col_offset,
                         sp.rows, sp.cols, sp.pattern});
                }
                continue;
            }
            for (size_t i : range(args.size())) for (size_t j : range(args.size())) {
                const auto fi = args[i]->field(), fj = args[j]->field();
                if (fi >= field::num_prim || fj >= field::num_prim || fi < fj ||
                    !is_active(args[i]) || !is_active(args[j])) continue;
                const auto &sp = f.hess_sparsity()[i][j];
                if (sp.pattern == sparsity::unknown) continue;
                add(target, fi, fj,
                    {get_expr_start_tangent(args[i]) + sp.row_offset,
                     get_expr_start_tangent(args[j]) + sp.col_offset,
                     sp.rows, sp.cols, sp.pattern});
            }
        }
    }
    for (auto &[key, profile_blocks] : blocks) {
        const auto target = static_cast<linear_target>(key / (field::num * field::num));
        const auto mode = target == linear_target::jacobian
                              ? sparse_plan_mode::distinct
                              : sparse_plan_mode::additive;
        linear_profile_.layouts.emplace(
            key, make_sparse_layout_plan(profile_blocks, mode));
    }

    if (tdim(__l) && dim(__dyn) && dim(__lift)) {
        const generic_dynamics *owner = nullptr;
        for (const generic_func &entry : exprs(__dyn)) {
            const auto *candidate =
                dynamic_cast<const generic_dynamics *>(&entry);
            if (!candidate || !candidate->owns_stage_elimination()) continue;
            if (owner)
                throw std::runtime_error(
                    "stage has multiple lifted elimination owners");
            owner = candidate;
        }
        if (!owner)
            throw std::runtime_error(
                "stage with explicit lifted variables requires one "
                "generic_dynamics elimination owner");
        for (const generic_func &entry : exprs(__lift)) {
            const auto *constraint =
                dynamic_cast<const generic_constr *>(&entry);
            if (!constraint || !owner->owns_subconstraint(*constraint))
                throw std::runtime_error(fmt::format(
                    "lifted constraint {} is not registered as a "
                    "sub-constraint of dynamics {}",
                    entry.name(), owner->name()));
        }
        for (const constr &constraint : owner->subconstraints()) {
            if (!contains(*constraint) || !is_active(*constraint) ||
                constraint->field() != __lift)
                throw std::runtime_error(fmt::format(
                    "dynamics {} sub-constraint {} must be an active "
                    "__lift constraint in the same stage",
                    owner->name(), constraint->name()));
        }

        auto program = std::make_shared<lifted_graph_program>();
        const auto jac = [&](field_t equation, field_t variable,
                             std::string name) {
            return symbolic_block(
                std::move(name), dim(equation), tdim(variable),
                linear_profile_.get(linear_target::jacobian, equation,
                                    variable), equation, variable, *program);
        };
        const auto residual = [&](field_t equation, std::string name) {
            cs::MX value = cs::MX::sym(
                std::move(name), static_cast<casadi_int>(dim(equation)), 1);
            program->inputs.push_back(value);
            program->input_layouts.emplace_back();
            program->input_bindings.push_back({
                lifted_graph_input_binding::kind::residual,
                equation, __undefined, 0, {}});
            return value;
        };
        std::map<std::array<size_t, 4>, lifted_symbolic_block>
            jacobian_blocks;
        lifted_symbolic_system system{
            .dyn_y = jac(__dyn, __y, "h_dyn_y"),
            .dyn_l = jac(__dyn, __l, "h_dyn_l"),
            .lift_y = jac(__lift, __y, "h_lift_y"),
            .lift_l = jac(__lift, __l, "h_lift_l"),
            .dyn_x = jac(__dyn, __x, "h_dyn_x"),
            .dyn_u = jac(__dyn, __u, "h_dyn_u"),
            .lift_x = jac(__lift, __x, "h_lift_x"),
            .lift_u = jac(__lift, __u, "h_lift_u"),
            .dyn_residual = residual(__dyn, "h_dyn_residual"),
            .lift_residual = residual(__lift, "h_lift_residual"),
            .action_rhs = cs::MX::sym(
                "h_lifted_action_rhs",
                static_cast<casadi_int>(tdim(__y) + tdim(__l)), 1),
        };
        system.jacobian_factory_ =
            [&, program](const cs::SX &raw_equation,
                         const sym &variable) -> lifted_symbolic_block {
            cs::SX equation = raw_equation;
            if (equation.size1() == 1 && equation.size2() != 1)
                equation = equation.T();
            if (equation.size2() != 1 || !equation.size1())
                throw std::invalid_argument(
                    "lifted Jacobian equation must be a nonempty vector");

            const generic_func *function = nullptr;
            size_t equation_offset = 0;
            for (const field_t field : std::array{__dyn, __lift}) {
                for (const generic_func &candidate : exprs(field)) {
                    const auto *task = candidate.get_codegen_task();
                    if (!task) continue;
                    for (size_t row = 0;
                         row + static_cast<size_t>(equation.size1()) <=
                         candidate.dim(); ++row) {
                        const cs::SX slice = task->sx_output(cs::Slice(
                            static_cast<casadi_int>(row),
                            static_cast<casadi_int>(
                                row + equation.size1())));
                        if (!cs::SX::is_equal(slice, equation)) continue;
                        if (function)
                            throw std::invalid_argument(
                                "lifted Jacobian equation is ambiguous");
                        function = &candidate;
                        equation_offset = row;
                    }
                }
            }
            if (!function)
                throw std::invalid_argument(
                    "lifted Jacobian equation is not a contiguous block of "
                    "the dynamics or lifted constraints");
            if (!function->has_arg(variable))
                return {cs::MX(casadi::Sparsity(
                            equation.size1(),
                            static_cast<casadi_int>(variable.tdim()))),
                        function->name() + "_" + variable.name() +
                            "_regularization",
                        &system};

            const std::array<size_t, 4> cache_key{
                function->uid(), equation_offset,
                static_cast<size_t>(equation.size1()), variable.uid()};
            if (const auto found = jacobian_blocks.find(cache_key);
                found != jacobian_blocks.end())
                return found->second;

            const size_t argument = function->arg_idx(variable);
            const size_t key = ocp_linear_profile::key(
                linear_target::jacobian, function->field(),
                variable.field());
            const auto source_it = jacobian_sources.find(key);
            const auto layout_it = linear_profile_.layouts.find(key);
            if (source_it == jacobian_sources.end() ||
                layout_it == linear_profile_.layouts.end())
                return {cs::MX(casadi::Sparsity(
                            equation.size1(),
                            static_cast<casadi_int>(variable.tdim()))),
                        function->name() + "_" + variable.name() +
                            "_regularization",
                        &system};

            const auto &sources = source_it->second;
            const auto &storage = layout_it->second;
            linear_backend::matrix_layout logical{
                static_cast<size_t>(equation.size1()), variable.tdim(), {}};
            std::vector<size_t> panel_indices;
            const auto panel_count = [&](sparsity pattern) {
                return static_cast<size_t>(std::ranges::count_if(
                    storage.panels, [=](const sparse_block_spec &panel) {
                        return panel.pattern == pattern;
                    }));
            };
            const size_t dense_panels = panel_count(sparsity::dense);
            const size_t diagonal_panels = panel_count(sparsity::diag);
            const auto physical_panel = [&](sparsity pattern,
                                            size_t index)
                -> const sparse_block_spec & {
                for (const auto &panel : storage.panels) {
                    if (panel.pattern != pattern) continue;
                    if (!index--) return panel;
                }
                throw std::logic_error(
                    "lifted Jacobian storage panel is missing");
            };
            const size_t equation_end =
                equation_offset + static_cast<size_t>(equation.size1());
            for (size_t source = 0; source < sources.size(); ++source) {
                const auto &entry = sources[source];
                if (entry.function->uid() != function->uid() ||
                    entry.argument != argument)
                    continue;
                const size_t begin =
                    std::max(equation_offset, entry.local.row_offset);
                const size_t end = std::min(
                    equation_end,
                    entry.local.row_offset + entry.local.rows);
                if (begin >= end) continue;
                const auto binding_it = std::ranges::find_if(
                    storage.bindings,
                    [=](const sparse_binding_spec &binding) {
                        return binding.source == source;
                    });
                if (binding_it == storage.bindings.end())
                    throw std::logic_error(
                        "lifted Jacobian block has no storage binding");
                const auto &binding = *binding_it;
                const auto &physical = physical_panel(
                    binding.storage_pattern, binding.panel);
                const size_t delta = begin - entry.local.row_offset;
                linear_backend::panel_layout panel{
                    entry.local.pattern,
                    begin - equation_offset,
                    entry.local.col_offset,
                    end - begin,
                    entry.local.pattern == sparsity::dense
                        ? entry.local.cols : end - begin};
                if (entry.local.pattern == sparsity::dense) {
                    panel.storage_offset =
                        binding.local_row + delta +
                        binding.local_col * physical.rows;
                    panel.storage_rows = physical.rows;
                } else {
                    panel.col_offset += delta;
                    panel.storage_offset = binding.local_row + delta;
                }
                logical.panels.push_back(panel);
                size_t pointer = binding.panel;
                if (binding.storage_pattern == sparsity::diag)
                    pointer += dense_panels;
                else if (binding.storage_pattern == sparsity::eye)
                    pointer += dense_panels + diagonal_panels;
                panel_indices.push_back(pointer);
            }
            if (logical.panels.empty())
                return {cs::MX(casadi::Sparsity(
                            equation.size1(),
                            static_cast<casadi_int>(variable.tdim()))),
                        function->name() + "_" + variable.name() +
                            "_regularization",
                        &system};

            const std::string name = fmt::format(
                "h_jac_{}_{}_{}_{}", function->name(), equation_offset,
                equation.size1(), variable.name());
            cs::MX input = cs::MX::sym(
                name, sparsity_from_layout(logical));
            program->inputs.push_back(input);
            program->input_layouts.push_back(std::move(logical));
            program->input_bindings.push_back({
                lifted_graph_input_binding::kind::jacobian_panel,
                function->field(), variable.field(), 0,
                std::move(panel_indices), function->uid(), variable.uid(), {}});
            lifted_symbolic_block result{
                std::move(input),
                function->name() + "_" + variable.name() +
                    "_regularization",
                &system};
            jacobian_blocks.emplace(cache_key, result);
            return result;
        };
        for (const auto equation : std::array{__dyn, __lift})
            for (const generic_func &function : exprs(equation))
                system.equations.push_back({
                    function.name(), equation, get_expr_start(function),
                    function.dim()});
        for (const auto field : primal_fields)
            for (const sym &variable : exprs(field))
                system.variables.push_back({
                    variable.name(), field,
                    get_expr_start_tangent(variable), variable.tdim()});

        const auto graph = owner->derive_elimination_graph(system);
        const auto graph_parameter = [&](const sym &parameter) {
            const std::string expected = "elim_param_" + parameter.name();
            const auto find_in = [&](const cs::MX &expression)
                -> std::optional<cs::MX> {
                for (const cs::MX &candidate : cs::MX::symvar(expression))
                    if (candidate.name() == expected)
                        return candidate;
                return std::nullopt;
            };
            for (const auto *expression :
                 {&graph.response_x, &graph.response_u,
                  &graph.response_residual, &graph.response_action})
                if (auto found = find_in(*expression)) return *found;
            for (const auto &intermediate : graph.intermediates)
                if (auto found = find_in(intermediate.value)) return *found;
            throw std::runtime_error(fmt::format(
                "lifted elimination parameter {} is not used by its graph",
                parameter.name()));
        };
        bool added_parameter = false;
        for (const var &parameter : owner->elimination_parameters()) {
            program->inputs.push_back(graph_parameter(*parameter));
            program->input_layouts.emplace_back();
            program->input_bindings.push_back({
                lifted_graph_input_binding::kind::parameter,
                __undefined, __undefined, 0, {}, 0, 0, parameter});
            if (contains(*parameter)) continue;
            if (!parameter->finalize())
                throw std::runtime_error(fmt::format(
                    "cannot finalize lifted elimination parameter {} uid {}",
                    parameter->name(), parameter->uid()));
            uids_.insert(parameter->uid());
            append_entry(parameter->handle());
            added_parameter = true;
        }
        if (added_parameter) rebuild_layout();
        std::set<std::string> used_graph_inputs;
        const auto collect_graph_inputs = [&](const cs::MX &expression) {
            for (const cs::MX &symbol : cs::MX::symvar(expression))
                used_graph_inputs.insert(symbol.name());
        };
        collect_graph_inputs(graph.response_x);
        collect_graph_inputs(graph.response_u);
        collect_graph_inputs(graph.response_residual);
        collect_graph_inputs(graph.response_action);
        for (const auto &intermediate : graph.intermediates)
            collect_graph_inputs(intermediate.value);
        for (const cs::MX &factor : *system.spd_factors_)
            collect_graph_inputs(factor);
        for (size_t i = program->inputs.size(); i-- > 0;) {
            if (program->inputs[i].is_symbolic() &&
                used_graph_inputs.contains(program->inputs[i].name()))
                continue;
            program->inputs.erase(program->inputs.begin() + i);
            program->input_layouts.erase(program->input_layouts.begin() + i);
            program->input_bindings.erase(
                program->input_bindings.begin() + i);
        }
        const size_t ny = tdim(__y), nl = tdim(__l);
        const size_t nh = ny + nl;
        const auto validate = [nh](const cs::MX &value, size_t cols,
                                   std::string_view name) {
            if (value.size1() != static_cast<casadi_int>(nh) ||
                value.size2() != static_cast<casadi_int>(cols))
                throw std::runtime_error(fmt::format(
                    "lifted elimination output {} must have shape ({}, {}), "
                    "got ({}, {})", name, nh, cols, value.size1(),
                    value.size2()));
        };
        validate(graph.response_x, tdim(__x), "response_x");
        validate(graph.response_u, tdim(__u), "response_u");
        validate(graph.response_residual, 1, "response_residual");
        if (graph.response_action.is_empty())
            throw std::runtime_error(
                "lifted elimination graph must provide response_action");
        validate(graph.response_action, 1, "response_action");
        program->response_x = graph.response_x;
        program->response_u = graph.response_u;
        program->response_residual = graph.response_residual;

        const auto plan_rows = [&](const cs::MX &value, field_t column) {
            const auto y = value(cs::Slice(0, static_cast<casadi_int>(ny)),
                                 cs::Slice());
            const auto l = value(
                cs::Slice(static_cast<casadi_int>(ny),
                          static_cast<casadi_int>(nh)), cs::Slice());
            linear_profile_.layouts[ocp_linear_profile::key(
                linear_target::lifted_projection, __y, column)] =
                layout_from_sparsity(y.sparsity());
            linear_profile_.layouts[ocp_linear_profile::key(
                linear_target::lifted_projection, __l, column)] =
                layout_from_sparsity(l.sparsity());
        };
        plan_rows(graph.response_x, __x);
        plan_rows(graph.response_u, __u);

        std::set<std::string> intermediate_names;
        for (const auto &intermediate : graph.intermediates) {
            if (intermediate.name.empty() ||
                !intermediate_names.insert(intermediate.name).second)
                throw std::runtime_error(
                    "lifted elimination intermediate names must be nonempty "
                    "and unique");
            linear_profile_.lifted_intermediates.push_back({
                intermediate.name,
                static_cast<size_t>(intermediate.value.size1()),
                static_cast<size_t>(intermediate.value.size2()),
                layout_from_sparsity(intermediate.value.sparsity())});
        }

        const auto append_response = [&](const cs::MX &value,
                                         field_t lifted,
                                         field_t column) {
            const size_t row = lifted == __y ? 0 : ny;
            const size_t rows = lifted == __y ? ny : nl;
            append_graph_outputs(
                value(cs::Slice(static_cast<casadi_int>(row),
                                static_cast<casadi_int>(row + rows)),
                      cs::Slice()),
                linear_profile_.get(linear_target::lifted_projection,
                                    lifted, column),
                program->projection_outputs);
        };
        append_response(graph.response_x, __y, __x);
        append_response(graph.response_x, __l, __x);
        append_response(graph.response_u, __y, __u);
        append_response(graph.response_u, __l, __u);
        for (size_t i = 0; i < graph.intermediates.size(); ++i)
            append_graph_outputs(
                graph.intermediates[i].value,
                linear_profile_.lifted_intermediates[i].layout,
                program->projection_outputs);
        program->projection_outputs.push_back(cs::MX::densify(
            graph.response_residual(
                cs::Slice(0, static_cast<casadi_int>(ny)), cs::Slice())));
        program->projection_outputs.push_back(cs::MX::densify(
            graph.response_residual(
                cs::Slice(static_cast<casadi_int>(ny),
                          static_cast<casadi_int>(nh)),
                cs::Slice())));
        program->action_rhs = system.action_rhs;
        program->action_output = cs::MX::densify(graph.response_action);
        program->transpose_rhs = cs::MX::sym(
            "h_lifted_transpose_rhs", static_cast<casadi_int>(nh), 1);
        program->transpose_output = cs::MX::densify(
            cs::MX::substitute(
                cs::MX::jtimes(graph.response_action, system.action_rhs,
                               program->transpose_rhs, true),
                system.action_rhs, cs::MX::zeros(nh, 1)));
        program->spd_factors = *system.spd_factors_;
        auto identity_inputs = program->inputs;
        identity_inputs.push_back(program->action_rhs);
        identity_inputs.push_back(program->transpose_rhs);
        auto identity_outputs = program->projection_outputs;
        identity_outputs.push_back(program->action_output);
        identity_outputs.push_back(program->transpose_output);
        identity_outputs.insert(identity_outputs.end(),
                                program->spd_factors.begin(),
                                program->spd_factors.end());
        const cs::Function identity_graph(
            "moto_lifted_graph_identity", identity_inputs,
            identity_outputs);
        program->artifact_identity =
            "lifted_graph_v1_" + utils::compute_md5_from_bytes(
                                     identity_graph.serialize());
        linear_profile_.lifted_program = std::move(program);
    }

}
void ocp_base::refresh_copy(const active_status_config &config) {
    finalized_ = false;
    if (!config.empty()) {
        update_active_status(config);
    }
}
void ocp_base::on_modified() {}
void ocp_base::maintain_order() {
    expr_list tmp;
    for (auto f : {__x, __y, __u}) {
        auto &syms = field_entries(f);
        tmp.reserve(syms.size());
        for (const generic_func &func : exprs(__dyn)) {
            const auto *dyn = dynamic_cast<const generic_dynamics *>(&func);
            for (const sym &arg : func.in_args(f)) {
                if (!is_active(arg)) continue;
                if (f == __u && (!dyn || dyn->input_shared(arg))) continue;
                auto it = std::find(syms.begin(), syms.end(), arg);
                if (it == syms.end()) {
                    throw std::runtime_error(fmt::format(
                        "order maintenance failure: "
                        "Dynamics {} arg {} uid {} not found in field {}",
                        func.name(), arg.name(), arg.uid(), f));
                }
                tmp.emplace_back(std::move(*it));
            }
        }
        std::erase_if(syms, [&](auto &&e) { return !e; });
        if (f != __u && !syms.empty()) {
            throw std::runtime_error(fmt::format(
                "order maintenance failure: "
                " field {} has exprs not in dynamics args",
                f));
        }
        std::ranges::move(syms, std::back_inserter(tmp));
        syms.swap(tmp);
    }
}
void ocp_base::print_summary() {
    finalize();
    fmt::print("-------------------------------------------------\n");
    fmt::print("problem uid {}\n", uid_);
    for (size_t i = 0; i < field::num; i++) {
        if (exprs(i).size() > 0) {
            fmt::print("field : {}, \ttotal dim {}, \ttotal tdim {}\n",
                       static_cast<field_t>(i),
                       dim(i), i < field::num_prim ? tdim(i) : 0);
            for (const auto &expr : exprs(i)) {
                fmt::print(" - {} uid {} dim: {} tdim: {}\n",
                           expr->name(), expr->uid(), expr->dim(), expr->tdim());
            }
        }
    }
    fmt::print("-------------------------------------------------\n");
}
void ocp_base::wait_until_ready() {
    for (const auto &f : field_entries()) {
        for (const auto &e : f) {
            if (!e->wait_until_ready()) {
                throw std::runtime_error(fmt::format(
                    "Expression {} with uid {} failed to be ready",
                    e->name(), e->uid()));
            }
        }
    }
    finalize();
}
bool ocp_base::accepts_term(const expr_handle &ex, std::string *reason) const {
    static_cast<void>(ex);
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}
ocp_ptr_t ocp::copy(const active_status_config &config) const {
    auto prob = ocp_ptr_t(new ocp(*this));
    prob->refresh_copy(config);
    return prob;
}

stage_ocp::stage_ocp(const stage_ocp &rhs)
    : ocp(rhs),
      endpoint_role_mask_by_uid_(rhs.endpoint_role_mask_by_uid_),
      mutation_revision_(rhs.mutation_revision()) {}

stage_ocp_ptr_t stage_ocp::copy(const active_status_config &config) const {
    auto prob = stage_ocp_ptr_t(new stage_ocp(*this));
    prob->refresh_copy(config);
    return prob;
}
bool stage_ocp::add_with_role(expr_handle ex, stage_expr_role role) {
    if (!ex) {
        throw std::runtime_error("Cannot add null expression to stage_ocp");
    }
    const size_t uid = ex->uid();
    const auto existing_it = endpoint_role_mask_by_uid_.find(uid);
    const bool had_existing = existing_it != endpoint_role_mask_by_uid_.end();
    const unsigned existing = had_existing ? existing_it->second : 0u;
    if (role == stage_expr_role::interval) {
        if (had_existing) {
            throw std::runtime_error(fmt::format(
                "Cannot add expression {} uid {} to both interval and endpoint stage roles",
                ex->name(), uid));
        }
        return ocp_base::add_impl(std::move(ex));
    }
    if (!had_existing && contains(*ex)) {
        throw std::runtime_error(fmt::format(
            "Cannot add expression {} uid {} to both interval and endpoint stage roles",
            ex->name(), uid));
    }
    const unsigned updated = existing | role_bit(role);
    endpoint_role_mask_by_uid_[uid] = updated;
    try {
        const bool added = ocp_base::add_impl(std::move(ex));
        if (!added && updated != existing) {
            on_modified();
        }
        return added;
    } catch (...) {
        if (!had_existing) {
            endpoint_role_mask_by_uid_.erase(uid);
        } else {
            endpoint_role_mask_by_uid_[uid] = existing;
        }
        throw;
    }
}
bool stage_ocp::validate_stage_term(const expr_handle &ex, std::string *reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!ex) {
        if (reason != nullptr) *reason = "null expression";
        return false;
    }
    const auto *func = as_generic_func(ex);
    if (func == nullptr) {
        return true;
    }
    if (is_pure_y_func(*func)) {
        if (reason != nullptr) {
            *reason = "pure y-only terms should be written on x and added through stage.ed.add(...)";
        }
        return false;
    }
    return true;
}
bool stage_ocp::validate_endpoint_term(const expr_handle &ex, std::string *reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (!ex) {
        if (reason != nullptr) *reason = "null expression";
        return false;
    }
    const auto *func = as_generic_func(ex);
    if (func == nullptr || ex->field() == __dyn || dynamic_cast<const generic_dynamics *>(ex.get()) != nullptr) {
        if (reason != nullptr) {
            *reason = "endpoint only accepts pure x cost/constraint terms; dynamics belong to stage.add(...)";
        }
        return false;
    }
    if (!func->has_pure_x_primal_args()) {
        if (reason != nullptr) {
            *reason = "endpoint only accepts terms with x/prm-style dependencies and no u or y arguments";
        }
        return false;
    }
    return true;
}
bool stage_ocp::accepts_term(const expr_handle &ex, std::string *reason) const {
    return validate_stage_term(ex, reason);
}
bool stage_ocp::has_role(const expr &ex, stage_expr_role role) const {
    if (auto it = endpoint_role_mask_by_uid_.find(ex.uid()); it != endpoint_role_mask_by_uid_.end()) {
        return (it->second & role_bit(role)) != 0u;
    }
    return role == stage_expr_role::interval;
}
void stage_ocp::set_mutation_callback(std::function<void()> callback) {
    mutation_callback_ = std::move(callback);
}
void stage_ocp::on_modified() {
    mutation_revision_.fetch_add(1, std::memory_order_release);
    if (mutation_callback_) {
        mutation_callback_();
    }
}
node_view stage_ocp::st() {
    return node_view(shared_from_this(), stage_expr_role::start_node);
}
node_view stage_ocp::ed() {
    return node_view(shared_from_this(), stage_expr_role::end_node);
}

node_view::node_view(const stage_ocp_ptr_t &stage, stage_expr_role role)
    : owner_(stage), role_(role) {}

void ocp_base::move_active_expr(const expr &ex, bool prune) {
    const size_t f = ex.field();
    auto &target_store = prune ? pruned_expr_ : disabled_expr_;
    auto &target_uids = prune ? pruned_uids_ : disabled_uids_;
    target_store[f].emplace_back(take_field_entry(ex));
    target_uids.insert(ex.uid());
    uids_.erase(ex.uid());
}
bool ocp_base::restore_inactive_expr(const expr &ex, bool from_pruned) {
    const size_t f = ex.field();
    auto &source_store = from_pruned ? pruned_expr_ : disabled_expr_;
    auto &source_uids = from_pruned ? pruned_uids_ : disabled_uids_;
    auto &source_list = source_store[f];
    if (find_entry(source_list, ex) == source_list.end()) {
        return false;
    }
    append_entry(take_entry(source_list, ex));
    uids_.insert(ex.uid());
    source_uids.erase(ex.uid());
    return true;
}
void ocp_base::update_active_status(const active_status_config &config) {
    for (expr &ex : config.activate_list) {
        if (!restore_inactive_expr(ex, true) && !restore_inactive_expr(ex, false)) {
            throw std::runtime_error(fmt::format("Cannot activate expression {} uid {}, it does not exist in the problem",
                                                 ex.name(), ex.uid()));
        }
    }
    for (const expr &ex : config.deactivate_list) {
        move_active_expr(ex, false);
    }
    for (int remaining = 5;; --remaining) {
        if (remaining == 0) {
            throw std::runtime_error("ocp::copy failed to converge during pruning");
        }
        bool changed = false;
        array_type<std::vector<std::reference_wrapper<const expr>>, status_func_fields> to_delete, to_re_enable;
        for (auto f : status_func_fields) {
            if (pruned_expr_[f].empty())
                continue;
            to_re_enable[f].reserve(pruned_expr_[f].size());
            for (const generic_func &e : pruned_expr_[f]) {
                if (can_be_active_by_status(e, this)) {
                    to_re_enable[f].emplace_back(e);
                }
            }
        }
        for (auto f : status_func_fields) {
            if (field_empty(f))
                continue;
            to_delete[f].reserve(field_entry_count(f));
            for (const generic_func &e : field_entries(f)) {
                if (!can_be_active_by_status(e, this)) {
                    to_delete[f].emplace_back(e);
                }
            }
        }
        for (auto f : status_func_fields) {
            for (const expr &e : to_delete[f]) {
                move_active_expr(e, true);
                if (std::find(config.activate_list.begin(), config.activate_list.end(), e) != config.activate_list.end()) {
                    throw std::runtime_error(fmt::format("func {} uid {} pruned but also in activate_list",
                                                         e.name(), e.uid()));
                }
                changed = true;
            }
            for (const expr &e : to_re_enable[f]) {
                restore_inactive_expr(e, true);
                if (std::find(config.deactivate_list.begin(), config.deactivate_list.end(), e) != config.deactivate_list.end()) {
                    throw std::runtime_error(fmt::format("func {} uid {} re-enabled but also in deactivate_list",
                                                         e.name(), e.uid()));
                }
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
    finalized_ = false;
    on_modified();
}
} // namespace moto
