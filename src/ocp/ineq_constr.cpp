#include <moto/ocp/ineq_constr.hpp>
#include <moto/solver/ipm/ipm_constr.hpp>
#include <moto/utils/codegen.hpp>

#include <limits>
#include <optional>

namespace moto {
namespace {

cs::SX as_column_vector(const cs::SX &sx) {
    if (sx.columns() == 1) {
        return sx;
    }
    std::vector<cs::SX> elems;
    elems.reserve(static_cast<size_t>(sx.numel()));
    for (casadi_int i = 0; i < sx.numel(); ++i) {
        elems.push_back(sx(i));
    }
    return cs::SX::vertcat(elems);
}

bool scalar_is_pos_inf(const cs::SX &sx) {
    return sx.is_constant() && sx.scalar().is_inf();
}

bool scalar_is_neg_inf(const cs::SX &sx) {
    return sx.is_constant() && sx.scalar().is_minus_inf();
}

std::shared_ptr<ineq_constr::box_spec> make_box_spec(const var_inarg_list &args,
                                                     const cs::SX &out,
                                                     const cs::SX &lb,
                                                     const cs::SX &ub) {
    auto spec = std::make_shared<ineq_constr::box_spec>();
    const casadi_int dim = out.numel();
    ineq_constr::box_side_array<cs::SX> bound;
    bound[box_side::ub] = ub;
    bound[box_side::lb] = lb;
    spec->base_dim = static_cast<size_t>(dim);
    for (auto side : box_sides) {
        spec->present_mask[side].resize(static_cast<Eigen::Index>(spec->base_dim));
        for (size_t i = 0; i < spec->base_dim; ++i) {
            const auto idx = static_cast<casadi_int>(i);
            spec->present_mask[side](static_cast<Eigen::Index>(i)) =
                side == box_side::ub ? !scalar_is_pos_inf(bound[side](idx)) : !scalar_is_neg_inf(bound[side](idx));
        }
        spec->has_side[side] = spec->present_mask[side].any();
    }
    const auto bind_symbolic_bound = [&](const cs::SX &bound_sx, box_side::side_t side) -> std::optional<var> {
        for (const sym &arg : args) {
            if (arg.field() < field::num_prim && cs::SX::depends_on(bound_sx, static_cast<const cs::SX &>(arg))) {
                throw std::runtime_error(side == box_side::lb ? "box lower bound must not depend on primal in_args"
                                                              : "box upper bound must not depend on primal in_args");
            }
        }
        bool all_entries_symbolic = true;
        for (const sym &arg : args) {
            if (arg.field() < field::num_prim || bound_sx.numel() != static_cast<casadi_int>(arg.dim())) {
                continue;
            }
            const cs::SX &arg_sx = static_cast<const cs::SX &>(arg);
            bool matches = true;
            for (casadi_int i = 0; i < bound_sx.numel(); ++i) {
                all_entries_symbolic = all_entries_symbolic && bound_sx(i).is_symbolic();
                if (!bound_sx(i).is_symbolic() || bound_sx(i).name() != arg_sx(i).name()) {
                    matches = false;
                }
            }
            if (matches) {
                return var(arg);
            }
        }
        if (all_entries_symbolic) {
            throw std::runtime_error(side == box_side::lb ? "box lower bound symbolic dependency is not listed in in_args"
                                                          : "box upper bound symbolic dependency is not listed in in_args");
        }
        throw std::runtime_error(side == box_side::lb ? "box lower bound must be constant or a direct non-primal in_arg"
                                                      : "box upper bound must be constant or a direct non-primal in_arg");
    };
    for (auto side : box_sides) {
        if (bound[side].is_constant()) {
            spec->bound_source[side] = ineq_constr::box_bound_source::constant;
            spec->bound_constant_value[side].resize(static_cast<Eigen::Index>(dim));
            for (casadi_int i = 0; i < dim; ++i) {
                spec->bound_constant_value[side](static_cast<Eigen::Index>(i)) = cs::DM(bound[side](i)).scalar();
            }
        } else {
            spec->bound_source[side] = ineq_constr::box_bound_source::in_arg;
            spec->bound_var[side] = *bind_symbolic_bound(bound[side], side);
        }
    }
    return spec;
}

} // namespace

constr ineq_constr::create(const std::string &name,
                           const var_inarg_list &args,
                           const cs::SX &out,
                           const box_bound_t &lb,
                           const box_bound_t &ub,
                           approx_order order,
                           field_t field) {
    const casadi_int dim = out.numel();
    const cs::SX out_vec = as_column_vector(out);
    const auto normalize_bound = [dim](const box_bound_t &bound, std::string_view which) {
        return std::visit(
            [&](const auto &value) -> cs::SX {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, scalar_t>) {
                    return cs::SX::ones(dim, 1) * value;
                } else if constexpr (std::is_same_v<T, vector>) {
                if (value.size() != dim) {
                    throw std::runtime_error(fmt::format("box {} bound has dim {}, expected {}", which, value.size(), dim));
                }
                    return cs::SX(cs::DM(std::vector<scalar_t>(value.data(), value.data() + value.size())));
                } else {
                    if (value.is_scalar()) {
                        return cs::SX::repmat(value, dim, 1);
                    }
                    if (value.numel() != dim) {
                        throw std::runtime_error(fmt::format("box {} bound has numel {}, expected {}", which, value.numel(), dim));
                    }
                    return as_column_vector(value);
                }
            },
            bound);
    };
    const cs::SX lb_sx = normalize_bound(lb, "lower");
    const cs::SX ub_sx = normalize_bound(ub, "upper");

    auto box = make_box_spec(args, out_vec, lb_sx, ub_sx);

    auto c = std::make_shared<solver::ipm_constr>(name, args, out_vec, order, field);
    c->set_box_info(std::move(box));
    return c;
}

constr ineq_constr::create(const std::string &name,
                           const var_inarg_list &args,
                           const cs::SX &out,
                           approx_order order,
                           field_t field) {
    auto c = std::make_shared<solver::ipm_constr>(name, args, out, order, field);
    return c;
}

constr ineq_constr::create(const std::string &name,
                           approx_order order,
                           size_t dim,
                           field_t field) {
    auto c = std::make_shared<solver::ipm_constr>(name, order, dim, field);
    return c;
}

} // namespace moto
