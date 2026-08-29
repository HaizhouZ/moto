#include <algorithm>
#include <cctype>

#include <moto/ocp/lifted.hpp>
#include <moto/ocp/impl/lag_data.hpp>
#include <moto/ocp/problem.hpp>
#include <moto/utils/codegen.hpp>



namespace moto {
namespace {
std::string parameter_component(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char c : value)
        result.push_back(std::isalnum(c) ? static_cast<char>(c) : '_');
    while (result.starts_with('_')) result.erase(result.begin());
    return result.empty() ? "block" : result;
}

std::string block_parameter_name(std::string_view equation,
                                 std::string_view variable) {
    return parameter_component(equation) + "_" +
           parameter_component(variable) + "_regularization";
}

bool matches_default(const sym &parameter,
                     const sym::default_val_t &default_value) {
    const vector &actual = parameter.default_value();
    if (std::holds_alternative<std::monostate>(default_value))
        return actual.size() == 0;
    if (const auto *scalar = std::get_if<scalar_t>(&default_value))
        return actual.size() == static_cast<Eigen::Index>(parameter.dim()) &&
               actual.isApprox(vector::Constant(parameter.dim(), *scalar));
    const vector &expected = std::get<vector>(default_value);
    return actual.size() == expected.size() && actual.isApprox(expected);
}

struct jacobian_piece {
    sp_info block;
    cs::SX value;
};

std::vector<jacobian_piece> split_jacobian_panels(const cs::SX &input) {
    const cs::SX expression = cs::SX::sparsify(input);
    const size_t rows = expression.rows(), cols = expression.columns();
    std::vector<unsigned char> used(rows * cols);
    const auto live = [&](size_t row, size_t col) {
        return !used[row + col * rows] && !expression(row, col).is_zero();
    };
    const auto mark = [&](size_t row, size_t col, size_t nr, size_t nc) {
        for (size_t j = col; j < col + nc; ++j)
            for (size_t i = row; i < row + nr; ++i)
                used[i + j * rows] = 1;
    };
    std::vector<jacobian_piece> result;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            if (!live(row, col)) continue;
            size_t nc = 0;
            while (col + nc < cols && live(row, col + nc)) ++nc;
            size_t nr = 1;
            while (row + nr < rows) {
                bool full = true;
                for (size_t j = 0; j < nc; ++j)
                    full &= live(row + nr, col + j);
                if (!full) break;
                ++nr;
            }
            if (nr < 2 || nc < 2) continue;
            mark(row, col, nr, nc);
            result.push_back({
                {sparsity::dense, row, col, nr, nc},
                expression(
                    cs::Slice(static_cast<casadi_int>(row),
                              static_cast<casadi_int>(row + nr)),
                    cs::Slice(static_cast<casadi_int>(col),
                              static_cast<casadi_int>(col + nc)))});
        }
    }
    for (ptrdiff_t offset = -static_cast<ptrdiff_t>(rows) + 1;
         offset < static_cast<ptrdiff_t>(cols); ++offset) {
        size_t row = offset < 0 ? -offset : 0;
        size_t col = offset > 0 ? offset : 0;
        while (row < rows && col < cols) {
            while (row < rows && col < cols && !live(row, col)) {
                ++row;
                ++col;
            }
            const size_t row_begin = row, col_begin = col;
            while (row < rows && col < cols && live(row, col)) {
                ++row;
                ++col;
            }
            if (row == row_begin) continue;
            const size_t size = row - row_begin;
            mark(row_begin, col_begin, size, size);
            cs::SX diagonal = cs::SX::diag(expression(
                cs::Slice(static_cast<casadi_int>(row_begin),
                          static_cast<casadi_int>(row)),
                cs::Slice(static_cast<casadi_int>(col_begin),
                          static_cast<casadi_int>(col))));
            const sparsity pattern = diagonal.is_one()
                                         ? sparsity::eye
                                         : sparsity::diag;
            result.push_back({
                {pattern, row_begin, col_begin, size, size},
                std::move(diagonal)});
        }
    }
    return result;
}

std::vector<scalar_t *> writable_panel_pointers(
    const sparse_matrix &source) {
    auto pointers = linear_backend::panel_pointers(source);
    const size_t diagonal = source.diagonal_segments_.empty()
                                ? source.diag_panels_.size()
                                : source.diagonal_segments_.size();
    pointers.resize(source.dense_panels_.size() + diagonal);
    return pointers;
}

void append_pointers(std::vector<scalar_t *> &destination,
                     const sparse_matrix &source) {
    auto pointers = writable_panel_pointers(source);
    destination.insert(destination.end(), pointers.begin(), pointers.end());
}

class generated_lifted final : public generic_dynamics {
  public:
    struct data final : generic_dynamics::approx_data {
        linear_backend::graph_kernel graph;
        std::vector<scalar_t *> graph_pointers;
        vector action_rhs, action_output;
        bool projection_ready = false;

        data(generic_constr::approx_data &&base,
             const generated_lifted &function);
    };

    generated_lifted(const generic_dynamics &source,
                     lifted_elimination_builder builder)
        : generic_dynamics(source), source_uid_(source.uid()) {
        install_elimination_graph(std::move(builder));
    }

    size_t elimination_source_uid() const override {
        return source_uid_;
    }

    func_approx_data_ptr_t create_approx_data(
        sym_data &primal, lag_data &raw, shared_data &shared) const override {
        generic_constr::approx_data base(
            func_approx_data(primal, raw, shared, *this));
        return std::make_unique<data>(std::move(base), *this);
    }
    std::span<const projection_panel> jacobian_panel_sparsity() const override {
        return jacobian_panels_;
    }
    void compute_project_jacobians(func_approx_data &) const override;
    void compute_project_residual(func_approx_data &) const override;
    void solve_stage_lifted_system(
        func_approx_data &, const matrix &, matrix &, bool) const override;

  protected:
    clone_ptr clone() const override { return new generated_lifted(*this); }
    void prepare_dynamics_codegen() override;
    void value_impl(func_approx_data &data) const override;
    void jacobian_impl(func_approx_data &data) const override;

  private:
    size_t source_uid_ = 0;
    std::vector<projection_panel> jacobian_panels_;
    void prepare_graph(data &) const;
    void update_projection(func_approx_data &) const;
};

generated_lifted::data::data(generic_constr::approx_data &&base,
                             const generated_lifted &function)
    : generic_dynamics::approx_data(std::move(base), true, true) {
    const auto &problem = *lag_data_->prob_;
    const size_t function_start = problem.get_expr_start(func_);
    jac_.clear();
    for (const auto &[argument, block] : function.jacobian_panels_) {
        const sym &symbol = func_.in_args(argument);
        jac_.push_back(approx_->jac_[symbol.field()].insert(
            function_start + block.row_offset,
            problem.get_expr_start_tangent(symbol) + block.col_offset,
            block.rows, block.cols, block.pattern));
    }

    const auto &program = problem.linear_profile().lifted_program;
    if (!program)
        throw std::logic_error(
            "generated lifted expression has no elimination program");
    std::vector<scalar_t *> inputs;
    inputs.reserve(program->input_bindings.size());
    for (const auto &binding : program->input_bindings) {
        switch (binding.source) {
        case lifted_graph_input_binding::kind::jacobian_panel: {
            const auto pointers = linear_backend::panel_pointers(
                lag_data_->approx_[binding.equation].jac_[binding.variable]);
            if (binding.panels.empty()) {
                inputs.insert(inputs.end(), pointers.begin(), pointers.end());
            } else {
                for (const size_t panel : binding.panels)
                    inputs.push_back(pointers.at(panel));
            }
            break;
        }
        case lifted_graph_input_binding::kind::residual:
            inputs.push_back(
                lag_data_->approx_[binding.equation].v_.data());
            break;
        case lifted_graph_input_binding::kind::parameter:
            inputs.push_back((*this)[*binding.parameter].data());
            break;
        }
    }
    if (program->action_output.is_empty() ||
        program->transpose_output.is_empty())
        throw std::logic_error(
            "lifted elimination graph requires forward and transpose actions");
    graph_pointers = inputs;
    action_rhs.setZero(problem.tdim(__y) + problem.tdim(__l));
    action_output.setZero(action_rhs.size());
    graph_pointers.push_back(action_rhs.data());
    graph_pointers.push_back(action_rhs.data());
    append_pointers(graph_pointers, dyn_proj_->proj_f_x_);
    append_pointers(graph_pointers, proj_l_x_);
    append_pointers(graph_pointers, dyn_proj_->proj_f_u_);
    append_pointers(graph_pointers, proj_l_u_);
    for (const auto &spec : problem.linear_profile().lifted_intermediates)
        append_pointers(graph_pointers,
                        lifted_intermediates_.at(spec.name));
    graph_pointers.push_back(dyn_proj_->proj_f_res_.data());
    graph_pointers.push_back(proj_l_res_.data());
    graph_pointers.push_back(action_output.data());
    graph_pointers.push_back(action_output.data());
}

void generated_lifted::prepare_dynamics_codegen() {
    auto *task = get_codegen_task();
    if (!task)
        throw std::runtime_error(
            "generated lifted expression requires a CasADi residual");
    jacobian_panels_.clear();
    task->jac_outputs.clear();
    const auto jacobian_for = [&](const sym &argument) {
        if (auto it = std::ranges::find_if(
                task->ext_jac, [&](const auto &entry) {
                    return entry.first->uid() == argument.uid();
                }); it != task->ext_jac.end())
            return it->second;
        return utils::cs_codegen::tangent_jacobian(
            task->sx_output, argument);
    };
    for (size_t i = 0; i < in_args_.size(); ++i) {
        const sym &argument = in_args_[i];
        if (!in_field(argument.field(), primal_fields)) continue;
        for (auto &[block, value] :
             split_jacobian_panels(jacobian_for(argument))) {
            jacobian_panels_.push_back({i, block});
            task->jac_outputs.push_back(std::move(value));
        }
    }
}

void generated_lifted::value_impl(func_approx_data &base) const {
    generic_dynamics::value_impl(base);
    base.as<data>().projection_ready = false;
}

void generated_lifted::jacobian_impl(func_approx_data &base) const {
    generic_dynamics::jacobian_impl(base);
    base.as<data>().projection_ready = false;
}

void generated_lifted::prepare_graph(data &local) const {
    if (local.graph) return;
    const auto &program =
        *local.lag_data_->prob_->linear_profile().lifted_program;
    auto graph_inputs = program.inputs;
    auto graph_input_layouts = program.input_layouts;
    graph_inputs.push_back(program.action_rhs);
    graph_input_layouts.emplace_back();
    graph_inputs.push_back(program.transpose_rhs);
    graph_input_layouts.emplace_back();
    const std::string artifact_identity =
        "lifted_projection_v1_" + program.artifact_identity;
    local.graph = linear_backend::compile_graph(
        artifact_identity, graph_inputs,
        std::vector<std::vector<cs::MX>>{
            program.projection_outputs,
            {program.action_output},
            {program.transpose_output}},
        graph_input_layouts,
        &local.lag_data_->lifted_graph_workspace_, "gen/linear_backend",
        program.spd_factors);
    if (local.graph_pointers.size() != local.graph.pointer_count() ||
        local.graph.entry_count() != 3)
        throw std::logic_error("lifted graph pointer layout mismatch");
}

void generated_lifted::update_projection(func_approx_data &base) const {
    auto &local = base.as<data>();
    if (!local.projection_ready) {
        prepare_graph(local);
        local.graph(0, local.graph_pointers);
        local.projection_ready = true;
    }
}

void generated_lifted::compute_project_jacobians(
    func_approx_data &base) const {
    update_projection(base);
}

void generated_lifted::compute_project_residual(
    func_approx_data &base) const {
    update_projection(base);
}

void generated_lifted::solve_stage_lifted_system(
    func_approx_data &base, const matrix &rhs, matrix &destination,
    bool use_transpose) const {
    auto &local = base.as<data>();
    prepare_graph(local);
    const size_t dimension = local.action_rhs.size();
    if (rhs.rows() != static_cast<Eigen::Index>(dimension))
        throw std::invalid_argument(
            "lifted graph action right-hand side row mismatch");
    destination.resize(dimension, rhs.cols());
    for (Eigen::Index column = 0; column < rhs.cols(); ++column) {
        local.action_rhs = rhs.col(column);
        local.graph(use_transpose ? 2 : 1, local.graph_pointers);
        destination.col(column) = local.action_output;
    }
}
} // namespace

var lifted_symbolic_block::param(sym::default_val_t default_value,
                                 std::string name, size_t dim) const {
    if (!system)
        throw std::logic_error("detached lifted symbolic block");
    if (name.empty()) name = parameter_name;
    return system->make_parameter(std::move(name), dim,
                                  std::move(default_value));
}

lifted_symbolic_block lifted_symbolic_block::rows(
    size_t begin, size_t end) const {
    if (begin > end || end > static_cast<size_t>(value.size1()))
        throw std::out_of_range(fmt::format(
            "lifted block row slice [{}, {}) exceeds {} rows",
            begin, end, value.size1()));
    return {value(cs::Slice(static_cast<casadi_int>(begin),
                            static_cast<casadi_int>(end)),
                  cs::Slice()),
            parameter_name + "_rows_" + std::to_string(begin) + "_" +
                std::to_string(end),
            system};
}

cs::MX lifted_symbolic_block::add_diag(const sym &parameter) const {
    if (!system)
        throw std::logic_error("detached lifted symbolic block");
    if (value.size1() != value.size2())
        throw std::invalid_argument(fmt::format(
            "cannot add diagonal regularization to a ({}, {}) block",
            value.size1(), value.size2()));
    const cs::MX symbolic = system->resolve_parameter(parameter);
    if (parameter.dim() == 1)
        return value + symbolic(0) * cs::MX::eye(value.size1());
    if (parameter.dim() == static_cast<size_t>(value.size1()))
        return value + cs::MX::diag(symbolic);
    throw std::invalid_argument(fmt::format(
        "diagonal regularization parameter {} must have dimension 1 or {}, "
        "got {}", parameter.name(), value.size1(), parameter.dim()));
}

cs::MX lifted_symbolic_factor::solve(const cs::MX &rhs) const {
    if (matrix.is_empty())
        throw std::logic_error("detached lifted linear factor");
    if (rhs.size1() != matrix.size1())
        throw std::invalid_argument(fmt::format(
            "linear factor right-hand side has {} rows, expected {}",
            rhs.size1(), matrix.size1()));
    return cs::MX::mtimes(inverse, rhs);
}

var lifted_symbolic_system::make_parameter(
    std::string name, size_t dim, sym::default_val_t default_value) const {
    if (!parameter_factory_)
        throw std::logic_error(
            "lifted symbolic system has no parameter factory");
    return parameter_factory_(std::move(name), dim,
                              std::move(default_value));
}

cs::MX lifted_symbolic_system::resolve_parameter(
    const sym &parameter) const {
    if (!parameter_resolver_)
        throw std::logic_error(
            "lifted symbolic system has no parameter resolver");
    if (auto it = parameter_by_uid_.find(parameter.uid());
        it != parameter_by_uid_.end())
        return it->second;
    auto [it, inserted] = parameter_by_uid_.emplace(
        parameter.uid(), parameter_resolver_(parameter));
    return it->second;
}

cs::MX lifted_symbolic_system::block_value(field_t equation,
                                           field_t variable) const {
    if (equation == __dyn && variable == __y) return dyn_y;
    if (equation == __dyn && variable == __l) return dyn_l;
    if (equation == __lift && variable == __y) return lift_y;
    if (equation == __lift && variable == __l) return lift_l;
    if (equation == __dyn && variable == __x) return dyn_x;
    if (equation == __dyn && variable == __u) return dyn_u;
    if (equation == __lift && variable == __x) return lift_x;
    if (equation == __lift && variable == __u) return lift_u;
    throw std::invalid_argument("invalid symbolic lifted-system block");
}

namespace {
const lifted_symbolic_partition &find_partition(
    const std::vector<lifted_symbolic_partition> &partitions,
    std::string_view name, std::string_view kind) {
    const lifted_symbolic_partition *result = nullptr;
    for (const auto &partition : partitions) {
        if (partition.name != name) continue;
        if (result)
            throw std::invalid_argument(fmt::format(
                "ambiguous lifted {} partition '{}'", kind, name));
        result = &partition;
    }
    if (!result)
        throw std::out_of_range(fmt::format(
            "unknown lifted {} partition '{}'", kind, name));
    return *result;
}

const lifted_symbolic_partition &find_partition(
    const std::vector<lifted_symbolic_partition> &partitions,
    size_t uid, std::string_view kind) {
    const auto found = std::ranges::find_if(
        partitions, [uid](const auto &partition) {
            return partition.uid == uid || partition.source_uid == uid;
        });
    if (found == partitions.end())
        throw std::out_of_range(fmt::format(
            "unknown lifted {} handle uid {}", kind, uid));
    return *found;
}
} // namespace

lifted_symbolic_block lifted_symbolic_system::block(
    field_t equation, field_t variable) const {
    return {block_value(equation, variable),
            block_parameter_name(field::name(equation),
                                 field::name(variable)),
            this};
}

lifted_symbolic_block lifted_symbolic_system::jac(
    const generic_func &equation, const sym &variable) const {
    const auto &row = find_partition(equations, equation.uid(), "equation");
    const auto &col = find_partition(variables, variable.uid(), "variable");
    return {block_value(row.field, col.field)(
        cs::Slice(static_cast<casadi_int>(row.offset),
                  static_cast<casadi_int>(row.offset + row.size)),
        cs::Slice(static_cast<casadi_int>(col.offset),
                  static_cast<casadi_int>(col.offset + col.size))),
            block_parameter_name(row.name, col.name), this};
}

lifted_symbolic_block lifted_symbolic_system::block(
    std::string_view equation, std::string_view variable) const {
    const auto &row = find_partition(equations, equation, "equation");
    const auto &col = find_partition(variables, variable, "variable");
    return {block_value(row.field, col.field)(
        cs::Slice(static_cast<casadi_int>(row.offset),
                  static_cast<casadi_int>(row.offset + row.size)),
        cs::Slice(static_cast<casadi_int>(col.offset),
                  static_cast<casadi_int>(col.offset + col.size))),
            block_parameter_name(equation, variable), this};
}

lifted_symbolic_block lifted_symbolic_system::jac(
    const cs::SX &equation, const sym &variable) const {
    if (!jacobian_factory_)
        throw std::logic_error(
            "lifted symbolic system has no Jacobian block factory");
    return jacobian_factory_(equation, variable);
}

cs::MX lifted_symbolic_system::residual(std::string_view equation) const {
    const auto &row = find_partition(equations, equation, "equation");
    const cs::MX &source = row.field == __dyn ? dyn_residual : lift_residual;
    return source(cs::Slice(static_cast<casadi_int>(row.offset),
                            static_cast<casadi_int>(row.offset + row.size)));
}

cs::MX lifted_symbolic_system::residual(
    const generic_func &equation) const {
    const auto &row = find_partition(equations, equation.uid(), "equation");
    const cs::MX &source = row.field == __dyn ? dyn_residual : lift_residual;
    return source(cs::Slice(static_cast<casadi_int>(row.offset),
                            static_cast<casadi_int>(row.offset + row.size)));
}

cs::MX lifted_symbolic_system::h_l() const {
    return cs::MX::vertcat(std::vector<cs::MX>{
        cs::MX::horzcat(std::vector<cs::MX>{dyn_y, dyn_l}),
        cs::MX::horzcat(std::vector<cs::MX>{lift_y, lift_l})});
}

cs::MX lifted_symbolic_system::h_x() const {
    return cs::MX::vertcat(std::vector<cs::MX>{dyn_x, lift_x});
}

cs::MX lifted_symbolic_system::h_u() const {
    return cs::MX::vertcat(std::vector<cs::MX>{dyn_u, lift_u});
}

cs::MX lifted_symbolic_system::h() const {
    return cs::MX::vertcat(
        std::vector<cs::MX>{dyn_residual, lift_residual});
}

lifted_symbolic_factor lifted_symbolic_system::solve(
    const cs::MX &matrix, bool spd) const {
    if (matrix.size1() != matrix.size2())
        throw std::invalid_argument(fmt::format(
            "linear factor requires a square matrix, got ({}, {})",
            matrix.size1(), matrix.size2()));
    if (!matrix.size1())
        throw std::invalid_argument(
            "linear factor requires a nonempty matrix");

    const cs::MX inverse = cs::MX::inv(matrix);
    if (spd) spd_factors_->push_back(inverse);
    return {matrix, inverse};
}

lifted_symbolic_projection lifted_symbolic_system::eliminate(
    const std::function<cs::MX(const cs::MX &)> &solve,
    std::vector<lifted_symbolic_intermediate> intermediates) const {
    if (!solve)
        throw std::invalid_argument(
            "lifted elimination solve callback is empty");
    const auto apply = [&](const cs::MX &rhs, std::string_view what) {
        cs::MX response = solve(rhs);
        if (response.size1() != rhs.size1() ||
            response.size2() != rhs.size2())
            throw std::invalid_argument(fmt::format(
                "lifted elimination solve returned {} shape ({}, {}), "
                "expected ({}, {})", what, response.size1(), response.size2(),
                rhs.size1(), rhs.size2()));
        return response;
    };
    const cs::MX hx = h_x(), hu = h_u(), residual = h();
    const cs::MX packed_rhs = cs::MX::horzcat(
        std::vector<cs::MX>{hx, hu, residual});
    const cs::MX packed_response = apply(packed_rhs, "projection");
    const casadi_int nx = hx.size2();
    const casadi_int nu = hu.size2();
    return {
        .response_x = packed_response(cs::Slice(), cs::Slice(0, nx)),
        .response_u = packed_response(
            cs::Slice(), cs::Slice(nx, nx + nu)),
        .response_residual = packed_response(
            cs::Slice(), cs::Slice(nx + nu, nx + nu + 1)),
        .intermediates = std::move(intermediates),
        .response_action = apply(action_rhs, "action"),
    };
}

implicit_lifted::implicit_lifted(const std::string &name, const cs::SX &out,
                                 const var_inarg_list &lifted_args,
                                 approx_order order)
    : generic_dynamics(name, out, order, __lift) {
    mark_lifted(lifted_args);
}

lifted implicit_lifted::create(const std::string &name, const cs::SX &out,
                               const var_inarg_list &lifted_args,
                               approx_order order) {
    return std::make_shared<implicit_lifted>(name, out, lifted_args, order);
}

void generic_dynamics::solve_stage_lifted_system(
    func_approx_data &, const matrix &, matrix &, bool) const {
    throw std::logic_error(
        "lifted stage solve requires an elimination graph action");
}

void generic_dynamics::compute_project_jacobians(
    func_approx_data &) const {
    throw std::logic_error(
        "lifted projection requires an elimination graph");
}

void generic_dynamics::compute_project_residual(
    func_approx_data &) const {
    throw std::logic_error(
        "lifted residual projection requires an elimination graph");
}

void generic_dynamics::apply_lifted_jacobian_inverse_transpose(
    func_approx_data &data, vector_ref v, vector_ref dst) const {
    if (!owns_stage_elimination())
        throw std::logic_error(
            "generic dynamics has no ordinary inverse-transpose implementation");
    const size_t ny = data.problem()->tdim(__y);
    const size_t nl = data.problem()->tdim(__l);
    matrix rhs = matrix::Zero(ny + nl, 1), solved;
    rhs.topRows(ny) = v;
    solve_stage_lifted_system(data, rhs, solved, true);
    dst = solved.topRows(ny);
}

void generic_dynamics::set_lifted_arguments(const var_inarg_list &args) {
    lifted_args_.clear();
    lifted_arg_uids_.clear();
    for (const sym &arg : args) {
        if (!in_field(arg.field(), lifted_fields))
            throw std::invalid_argument(fmt::format(
                "Lifted argument {} of {} must be in a lifted field, got field {}",
                arg.name(), name(), arg.field()));
        if (!lifted_arg_uids_.insert(arg.uid()).second)
            continue;
        lifted_args_.push_back(arg);
    }
}

void generic_dynamics::mark_lifted(const var_inarg_list &args) {
    field_write_guard();
    for (const sym &arg : args) {
        add_argument(arg);
        // A lifted column may intentionally be absent from the authored
        // residual and introduced only by the elimination graph, e.g. a
        // Delassus regularization block.
        skip_unused_arg_check_.insert(arg.uid());
    }
    var_list merged = lifted_args_;
    merged.insert(merged.end(), args.begin(), args.end());
    set_lifted_arguments(merged);
}

bool generic_dynamics::is_lifted(const sym &arg) const {
    return lifted_arg_uids_.contains(arg.uid());
}

size_t generic_dynamics::lifted_tdim() const {
    size_t result = 0;
    for (const sym &arg : lifted_args_)
        result += arg.tdim();
    return result;
}

void generic_dynamics::add_subconstraint(const constr &constraint) {
    field_write_guard();
    if (!constraint)
        throw std::invalid_argument("dynamics sub-constraint is null");
    if (constraint.get() == this)
        throw std::invalid_argument(
            "dynamics cannot contain itself as a sub-constraint");
    if (std::ranges::any_of(subconstraints_, [&](const constr &existing) {
            return existing->uid() == constraint->uid();
        }))
        return;
    subconstraints_.push_back(constraint);
    add_dep(constraint);
}

bool generic_dynamics::owns_subconstraint(
    const generic_constr &constraint) const {
    return std::ranges::any_of(subconstraints_, [&](const constr &entry) {
        return entry->uid() == constraint.uid();
    });
}

void generic_dynamics::install_elimination_graph(
    lifted_elimination_builder builder) {
    field_write_guard();
    if (!builder)
        throw std::invalid_argument("lifted elimination graph builder is empty");
    elimination_builder_ = std::move(builder);
    elimination_parameter_state_ =
        std::make_shared<elimination_parameter_state>();
}

lifted generic_dynamics::set_elimination_graph(
    lifted_elimination_builder builder) const {
    return with_elimination_graph(std::move(builder));
}

lifted generic_dynamics::with_elimination_graph(
    lifted_elimination_builder builder,
    const std::vector<constr> &subconstraints) const {
    if (!builder)
        throw std::invalid_argument("lifted elimination graph builder is empty");
    auto result =
        std::make_shared<generated_lifted>(*this, std::move(builder));
    for (const constr &constraint : subconstraints)
        result->add_subconstraint(constraint);
    return result;
}

var generic_dynamics::get_or_create_elimination_parameter(
    std::string name, size_t dim,
    sym::default_val_t default_value) const {
    if (name.empty())
        throw std::invalid_argument(
            "lifted elimination parameter name must be nonempty");
    if (!dim)
        throw std::invalid_argument(
            "lifted elimination parameter dimension must be nonzero");
    std::lock_guard lock(elimination_parameter_state_->mutex);
    for (const var &parameter : elimination_parameter_state_->parameters) {
        if (parameter->name() != name) continue;
        if (parameter->dim() != dim ||
            !matches_default(*parameter, default_value))
            throw std::invalid_argument(fmt::format(
                "lifted elimination parameter '{}' was requested with "
                "inconsistent dimension or default value", name));
        return parameter;
    }
    var parameter = sym::params(name, dim, std::move(default_value));
    elimination_parameter_state_->parameters.push_back(parameter);
    return parameter;
}

cs::MX generic_dynamics::use_elimination_parameter(
    const sym &parameter) const {
    if (parameter.field() != __p)
        throw std::invalid_argument(fmt::format(
            "lifted elimination parameter {} must use field __p, got {}",
            parameter.name(), field::name(parameter.field())));
    {
        std::lock_guard lock(elimination_parameter_state_->mutex);
        const auto &parameters = elimination_parameter_state_->parameters;
        if (auto it = std::ranges::find_if(
                parameters, [&](const var &existing) {
                    return existing->uid() == parameter.uid();
                });
            it == parameters.end()) {
            if (std::ranges::any_of(parameters, [&](const var &existing) {
                    return existing->name() == parameter.name();
                }))
                throw std::invalid_argument(fmt::format(
                    "lifted elimination parameter name '{}' refers to "
                    "multiple symbols", parameter.name()));
            elimination_parameter_state_->parameters.push_back(
                expr_cast<sym>(parameter.handle()));
        }
    }
    return cs::MX::sym("elim_param_" + parameter.name(),
                       static_cast<casadi_int>(parameter.dim()), 1);
}

lifted_symbolic_projection generic_dynamics::derive_elimination_graph(
    const lifted_symbolic_system &system) const {
    system.parameter_factory_ =
        [this](std::string name, size_t dim,
               sym::default_val_t default_value) {
            return get_or_create_elimination_parameter(
                std::move(name), dim, std::move(default_value));
        };
    system.parameter_resolver_ = [this](const sym &parameter) {
        return use_elimination_parameter(parameter);
    };
    system.parameter_by_uid_.clear();
    system.spd_factors_->clear();
    if (!elimination_builder_)
        throw std::logic_error(fmt::format(
            "dynamics {} requires an explicit lifted elimination graph",
            name()));
    return elimination_builder_(system);
}

void generic_dynamics::substitute(const sym &arg, const sym &rhs) {
    generic_constr::substitute(arg, rhs);
    {
        std::lock_guard lock(elimination_parameter_state_->mutex);
        auto &parameters = elimination_parameter_state_->parameters;
        if (auto it = std::find(parameters.begin(), parameters.end(), arg);
            it != parameters.end()) {
            if (!elimination_parameter_state_.unique()) {
                auto detached =
                    std::make_shared<elimination_parameter_state>();
                detached->parameters = parameters;
                elimination_parameter_state_ = std::move(detached);
                it = std::find(elimination_parameter_state_->parameters.begin(),
                               elimination_parameter_state_->parameters.end(),
                               arg);
            }
            *it = rhs;
            skip_unused_arg_check_.erase(arg.uid());
            skip_unused_arg_check_.insert(rhs.uid());
        }
    }
    if (is_lifted(arg)) {
        std::replace(lifted_args_.begin(), lifted_args_.end(), arg, rhs);
        lifted_arg_uids_.erase(arg.uid());
        lifted_arg_uids_.insert(rhs.uid());
        skip_unused_arg_check_.erase(arg.uid());
        skip_unused_arg_check_.insert(rhs.uid());
    }
    if (input_shared(arg)) {
        std::replace(shared_inputs_.begin(), shared_inputs_.end(), arg, rhs);
        shared_inputs_indices_.erase(arg.uid());
        shared_inputs_indices_.insert(rhs.uid());
    }
}

void generic_dynamics::finalize_impl() {
    const bool dynamics_group = field() != __lift;
    if (dynamics_group) {
        var_list predicted_states;
        for (const sym &arg : in_args_)
            if (arg.field() == __y)
                predicted_states.push_back(arg);
        set_lifted_arguments(predicted_states);

        var_list reordered;
        reordered.reserve(shared_inputs_.size());
        for (const sym &s : shared_inputs_)
            if (auto it = std::find(in_args_.begin(), in_args_.end(), s);
                it != in_args_.end())
                reordered.emplace_back(std::move(*it));
        std::erase_if(in_args_, [](const auto &arg) { return !arg; });
        for (auto &arg : reordered)
            in_args_.emplace_back(std::move(arg));

    }

    if (lifted_args_.empty())
        throw std::runtime_error(fmt::format(
            "Dynamics/lifting group {} has no lifted arguments", name()));
    for (const sym &arg : lifted_args_) {
        if (std::find(in_args_.begin(), in_args_.end(), arg) == in_args_.end())
            throw std::runtime_error(fmt::format(
                "Lifted argument {} is not an argument of {}",
                arg.name(), name()));
    }
    if (lifted_tdim() != dim())
        throw std::runtime_error(fmt::format(
            "Dynamics/lifting group {} requires a square lifted Jacobian: "
            "lifted tangent dimension={}, residual dimension={}",
            name(), lifted_tdim(), dim()));

    if (dynamics_group) prepare_dynamics_codegen();
    generic_constr::finalize_impl();

    if (dynamics_group) {
        var_list reordered;
        shared_inputs_indices_.clear();
        for (var &s : shared_inputs_)
            if (has_arg(s)) {
                reordered.emplace_back(std::move(s));
                shared_inputs_indices_.insert(reordered.back()->uid());
            }
        shared_inputs_.swap(reordered);
    }
}

} // namespace moto
