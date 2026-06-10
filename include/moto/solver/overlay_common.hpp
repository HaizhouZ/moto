#pragma once

#include <moto/ocp/constr.hpp>
#include <moto/ocp/impl/node_data.hpp>

#include <array>
#include <string_view>

namespace moto::solver::overlay {

inline std::string overlay_name(const generic_func &source, std::string_view suffix) {
    return fmt::format("{}__{}", source.name(), suffix);
}

inline void copy_source_sparsity(generic_func &dst, const generic_func &src) {
    const auto &src_args = src.in_args();
    const auto &src_sp = src.jac_sparsity();
    for (size_t i = 0; i < src_args.size() && i < src_sp.size(); ++i) {
        dst.set_jac_sparsity(src_args[i], src_sp[i]);
    }
    dst.set_hess_sparsity(src.hess_sparsity());
}

template <typename Factory, size_t N>
void add_constr_overlay_group(const ocp_ptr_t &source_prob,
                              ocp_ptr_t &overlay_prob,
                              const std::array<field_t, N> &fields,
                              Factory &&factory) {
    for (auto field : fields) {
        for (const shared_expr &expr : source_prob->exprs(field)) {
            if (auto source = std::dynamic_pointer_cast<generic_constr>(expr)) {
                overlay_prob->add(*factory(constr(std::move(source))));
            }
        }
    }
}

inline void copy_primal_and_params(const node_data &src, node_data &dst) {
    for (auto field : primal_fields) {
        dst.sym_val().value_[field] = src.sym_val().value_[field];
    }
    dst.sym_val().value_[__p] = src.sym_val().value_[__p];
}

inline void copy_dense_dual_if_present(const node_data &src, node_data &dst, field_t field) {
    if (src.dense().dual_[field].size() > 0 && dst.dense().dual_[field].size() > 0) {
        dst.dense().dual_[field] = src.dense().dual_[field];
    }
}

template <size_t N>
void copy_dense_duals_if_present(const node_data &src,
                                 node_data &dst,
                                 const std::array<field_t, N> &fields) {
    for (auto field : fields) {
        copy_dense_dual_if_present(src, dst, field);
    }
}

template <typename Overlay>
void copy_source_multiplier(vector_ref dst, const node_data &src, const Overlay &overlay) {
    const auto &source_data = src.data(overlay.source());
    dst = const_cast<func_approx_data &>(source_data).template as<generic_constr::approx_data>().multiplier_;
}

template <typename Overlay>
void commit_source_multiplier(node_data &dst, const Overlay &overlay, const vector_const_ref &src) {
    auto &source_data = dst.data(overlay.source());
    source_data.template as<generic_constr::approx_data>().multiplier_ = src;
}

template <typename Overlay, typename Fn>
void for_each_overlay(node_data &node, field_t field, Fn &&fn) {
    node.for_each(field, [&](const Overlay &overlay, typename Overlay::approx_data &data) {
        fn(overlay, data);
    });
}

template <typename Overlay, typename Fields, typename Fn>
void for_each_overlay_field(node_data &node, const Fields &fields, Fn &&fn) {
    for (auto field : fields) {
        for_each_overlay<Overlay>(node, field, fn);
    }
}

template <typename Overlay, typename Fields>
void copy_source_multipliers(node_data &source, node_data &overlay_node, const Fields &fields) {
    for_each_overlay_field<Overlay>(overlay_node, fields, [&](const Overlay &overlay, typename Overlay::approx_data &data) {
        copy_source_multiplier(data.multiplier_, source, overlay);
    });
}

template <typename Overlay, typename Fields>
void commit_source_multipliers(node_data &source, node_data &overlay_node, const Fields &fields) {
    for_each_overlay_field<Overlay>(overlay_node, fields, [&](const Overlay &overlay, typename Overlay::approx_data &data) {
        commit_source_multiplier(source, overlay, data.multiplier_);
    });
}

enum class pair_match { same_size, prefix };

template <pair_match Match, typename Fn>
void for_each_overlay_pair(node_data &source,
                           node_data &overlay,
                           field_t field,
                           std::string_view context,
                           Fn &&fn) {
    const auto &source_exprs = source.problem().exprs(field);
    const auto &overlay_exprs = overlay.problem().exprs(field);
    const bool mismatch = Match == pair_match::same_size
                              ? source_exprs.size() != overlay_exprs.size()
                              : overlay_exprs.size() < source_exprs.size();
    if (mismatch) {
        throw std::runtime_error(fmt::format("{} {} source/overlay {} mismatch: {} vs {}",
                                             context,
                                             field::name(field),
                                             Match == pair_match::same_size ? "size" : "prefix",
                                             source_exprs.size(),
                                             overlay_exprs.size()));
    }
    for (size_t i = 0; i < source_exprs.size(); ++i) {
        fn(constr(source_exprs[i]), constr(overlay_exprs[i]));
    }
}

} // namespace moto::solver::overlay
