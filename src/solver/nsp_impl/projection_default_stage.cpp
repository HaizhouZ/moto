#include <moto/solver/projection/default_stage.hpp>
#include <moto/solver/ns_riccati/ns_riccati_data.hpp>

#include <algorithm>
#include <fmt/core.h>

namespace moto::solver::projection {

namespace {

std::string block_group_name(const std::string &base, size_t index, size_t total) {
    if (total <= 1) {
        return base;
    }
    return fmt::format("{}:{}", base, index);
}

void append_field_blocks(std::vector<compiled_equality_block> &blocks,
                         equality_block_kind kind,
                         size_t total_rows,
                         const std::vector<size_t> &parts,
                         size_t &row_begin,
                         const std::string &group) {
    size_t source_begin = 0;
    size_t block_index = 0;
    for (size_t part : parts) {
        if (part == 0) {
            continue;
        }
        blocks.push_back(compiled_equality_block{
            .kind = kind,
            .source_begin = source_begin,
            .source_count = part,
            .row_begin = row_begin,
            .group = block_group_name(group, block_index++, parts.size()),
        });
        source_begin += part;
        row_begin += part;
    }
    if (source_begin != total_rows) {
        throw std::runtime_error("compiled_default_elimination_stage partition does not cover the source rows");
    }
}

void rebuild_layout(compiled_default_elimination_stage &stage, size_t nu) {
    stage.rows.clear();
    for (const auto &block : stage.blocks) {
        const auto kind = block.kind == equality_block_kind::projected_state
                              ? row_kind::projected_state_eq
                              : row_kind::state_input_eq;
        for (size_t i = 0; i < block.source_count; ++i) {
            stage.rows.push_back(row_layout{
                .kind = kind,
                .source_index = static_cast<uint32_t>(block.source_begin + i),
                .canonical_index = static_cast<uint32_t>(block.row_begin + i),
                .group = block.group,
            });
        }
    }

    stage.cols.clear();
    stage.cols.reserve(nu);
    for (uint32_t j = 0; j < nu; ++j) {
        stage.cols.push_back(column_layout{
            .kind = column_kind::control,
            .source_index = j,
            .canonical_index = j,
            .group = "__u",
        });
    }
}

void copy_sparse_row_slice(const sparse_mat &src,
                           size_t src_row_begin,
                           size_t src_row_count,
                           matrix_ref dst) {
    if (dst.rows() != static_cast<Eigen::Index>(src_row_count) ||
        dst.cols() != static_cast<Eigen::Index>(src.cols())) {
        throw std::runtime_error("copy_sparse_row_slice destination shape mismatch");
    }
    dst.setZero();

    const int src_begin = static_cast<int>(src_row_begin);
    const int src_end = static_cast<int>(src_row_begin + src_row_count);

    auto copy_dense_panel = [&](const auto &panel) {
        const int overlap_begin = std::max(panel.row_st_, src_begin);
        const int overlap_end = std::min(panel.row_ed_, src_end);
        if (overlap_begin >= overlap_end) {
            return;
        }
        const int panel_row_begin = overlap_begin - panel.row_st_;
        const int out_row_begin = overlap_begin - src_begin;
        const int overlap_rows = overlap_end - overlap_begin;
        dst.block(out_row_begin, panel.col_st_, overlap_rows, panel.cols_) =
            panel.data_.middleRows(panel_row_begin, overlap_rows);
    };

    for (const auto &panel : src.dense_panels_) {
        copy_dense_panel(panel);
    }

    for (const auto &panel : src.diag_panels_) {
        const int overlap_begin = std::max(panel.row_st_, src_begin);
        const int overlap_end = std::min(panel.row_ed_, src_end);
        if (overlap_begin >= overlap_end) {
            continue;
        }
        const int panel_row_begin = overlap_begin - panel.row_st_;
        const int out_row_begin = overlap_begin - src_begin;
        const int overlap_rows = overlap_end - overlap_begin;
        dst.block(out_row_begin,
                  panel.col_st_ + panel_row_begin,
                  overlap_rows,
                  overlap_rows)
            .diagonal() = panel.data_.segment(panel_row_begin, overlap_rows);
    }

    for (const auto &panel : src.eye_panels_) {
        const int overlap_begin = std::max(panel.row_st_, src_begin);
        const int overlap_end = std::min(panel.row_ed_, src_end);
        if (overlap_begin >= overlap_end) {
            continue;
        }
        const int panel_row_begin = overlap_begin - panel.row_st_;
        const int out_row_begin = overlap_begin - src_begin;
        const int overlap_rows = overlap_end - overlap_begin;
        dst.block(out_row_begin,
                  panel.col_st_ + panel_row_begin,
                  overlap_rows,
                  overlap_rows)
            .diagonal()
            .setOnes();
    }
}

bool has_projected_state_block(const compiled_default_elimination_stage &stage) {
    return std::any_of(stage.blocks.begin(), stage.blocks.end(), [](const compiled_equality_block &block) {
        return block.kind == equality_block_kind::projected_state && block.source_count > 0;
    });
}

} // namespace

void compiled_default_elimination_stage::configure(size_t ns, size_t nc, size_t nu) {
    std::vector<size_t> projected_state_parts;
    std::vector<size_t> state_input_parts;
    if (ns > 0) {
        projected_state_parts.push_back(ns);
    }
    if (nc > 0) {
        state_input_parts.push_back(nc);
    }
    configure_partitioned(ns, nc, nu, projected_state_parts, state_input_parts);
}

void compiled_default_elimination_stage::configure_partitioned(size_t ns, size_t nc, size_t nu,
                                                               const std::vector<size_t> &projected_state_parts,
                                                               const std::vector<size_t> &state_input_parts) {
    projected_state_rows = ns;
    state_input_rows = nc;

    blocks.clear();
    size_t row_begin = 0;
    append_field_blocks(blocks, equality_block_kind::projected_state, ns, projected_state_parts, row_begin, "__eq_x_projected");
    append_field_blocks(blocks, equality_block_kind::state_input, nc, state_input_parts, row_begin, "__eq_xu");

    rebuild_layout(*this, nu);
}

void compiled_default_elimination_stage::configure_blocks(size_t ns, size_t nc, size_t nu,
                                                          const std::vector<equality_block_spec> &block_specs) {
    projected_state_rows = ns;
    state_input_rows = nc;

    blocks.clear();
    size_t row_begin = 0;
    size_t covered_projected_state = 0;
    size_t covered_state_input = 0;

    for (const auto &spec : block_specs) {
        if (spec.source_count == 0) {
            continue;
        }

        const size_t source_end = spec.source_begin + spec.source_count;
        const size_t limit = spec.kind == equality_block_kind::projected_state ? ns : nc;
        if (source_end > limit) {
            throw std::runtime_error("compiled_default_elimination_stage block exceeds source range");
        }

        auto &covered = spec.kind == equality_block_kind::projected_state ? covered_projected_state : covered_state_input;
        if (spec.source_begin != covered) {
            throw std::runtime_error("compiled_default_elimination_stage blocks must cover each field contiguously without gaps");
        }

        blocks.push_back(compiled_equality_block{
            .kind = spec.kind,
            .source_begin = spec.source_begin,
            .source_count = spec.source_count,
            .row_begin = row_begin,
            .group = spec.group,
        });
        covered = source_end;
        row_begin += spec.source_count;
    }

    if (covered_projected_state != ns || covered_state_input != nc) {
        throw std::runtime_error("compiled_default_elimination_stage block configuration does not cover all equality rows");
    }

    rebuild_layout(*this, nu);
}

void compiled_default_elimination_stage::prepare_factorization_problem(ns_riccati::ns_riccati_data &d) const {
    auto &nsp = d.nsp_;

    d.nis = 0;
    d.nic = 0;
    d.ncstr = row_count();

    nsp.s_c_stacked.conservativeResize(d.ncstr, d.nu);
    nsp.s_c_stacked.setZero();
    nsp.s_c_stacked_0_K.conservativeResize(d.ncstr, d.nx);
    nsp.s_c_stacked_0_K.setZero();
    d.d_lbd_s_c.conservativeResize(d.ncstr);
}

void compiled_default_elimination_stage::prepare_residual_problem(ns_riccati::ns_riccati_data &d) const {
    d.ncstr = row_count();
    d.nsp_.s_c_stacked_0_k.conservativeResize(d.ncstr);
}

void compiled_default_elimination_stage::assemble_control_jacobian(ns_riccati::ns_riccati_data &d) const {
    auto &nsp = d.nsp_;

    if (has_projected_state_block(*this)) {
        nsp.s_u.conservativeResize(projected_state_rows, d.nu);
        nsp.s_u.setZero();
        d.s_y.times<false>(d.F_u, nsp.s_u);
    }

    for (const auto &block : blocks) {
        auto dst = nsp.s_c_stacked.middleRows(block.row_begin, block.source_count);
        if (block.kind == equality_block_kind::projected_state) {
            dst = nsp.s_u.middleRows(block.source_begin, block.source_count);
        } else {
            copy_sparse_row_slice(d.c_u, block.source_begin, block.source_count, dst);
        }
    }
}

void compiled_default_elimination_stage::assemble_state_jacobian(ns_riccati::ns_riccati_data &d) const {
    auto &nsp = d.nsp_;

    if (has_projected_state_block(*this)) {
        nsp.s_0_p_K.conservativeResize(projected_state_rows, d.nx);
        nsp.s_0_p_K.setZero();
        d.s_x.dump_into(nsp.s_0_p_K);
        d.s_y.times<false>(d.F_x, nsp.s_0_p_K);
    }

    for (const auto &block : blocks) {
        auto dst = nsp.s_c_stacked_0_K.middleRows(block.row_begin, block.source_count);
        if (block.kind == equality_block_kind::projected_state) {
            dst = nsp.s_0_p_K.middleRows(block.source_begin, block.source_count);
        } else {
            copy_sparse_row_slice(d.c_x, block.source_begin, block.source_count, dst);
        }
    }
}

void compiled_default_elimination_stage::assemble_residual(ns_riccati::ns_riccati_data &d) const {
    auto &nsp = d.nsp_;
    auto &_approx = d.dense_->approx_;

    if (has_projected_state_block(*this)) {
        nsp.s_0_p_k.conservativeResize(projected_state_rows);
        nsp.s_0_p_k.noalias() = _approx[__eq_x].v_;
        d.s_y.times<false>(d.F_0, nsp.s_0_p_k);
    }

    for (const auto &block : blocks) {
        auto dst = nsp.s_c_stacked_0_k.segment(block.row_begin, block.source_count);
        if (block.kind == equality_block_kind::projected_state) {
            dst = nsp.s_0_p_k.segment(block.source_begin, block.source_count);
        } else {
            dst = _approx[__eq_xu].v_.segment(block.source_begin, block.source_count);
        }
    }
}

void compiled_default_elimination_stage::compose_particular_and_nullspace(const ns_riccati::ns_riccati_data &d,
                                                                          const matrix &rhs,
                                                                          matrix &particular,
                                                                          matrix &basis) const {
    const auto &A = d.nsp_.s_c_stacked;
    if (rhs.rows() != A.rows()) {
        throw std::runtime_error("compose_particular_and_nullspace rhs row mismatch");
    }
    // For the dense reference path, block order changes the assembled operator and therefore the
    // factorization layout, but the actual particular/nullspace solve still uses the full stacked
    // system. Naive sequential block elimination is not equivalent for arbitrary block orders.
    particular = d.nsp_.lu_eq_.solve(rhs);
    const auto nullity = static_cast<Eigen::Index>(d.nsp_.lu_eq_.dimensionOfKernel());
    basis = d.nsp_.lu_eq_.kernel().leftCols(nullity);
}

void compiled_default_elimination_stage::compose_particular(const ns_riccati::ns_riccati_data &d,
                                                            const vector &rhs,
                                                            vector &particular) const {
    if (rhs.rows() != d.nsp_.s_c_stacked.rows()) {
        throw std::runtime_error("compose_particular rhs row mismatch");
    }
    particular = d.nsp_.lu_eq_.solve(rhs);
}

void compiled_default_elimination_stage::export_problem(const ns_riccati::ns_riccati_data &d,
                                                        assembled_problem &prob) const {
    auto &nsp = d.nsp_;
    prob.nx = d.nx;
    prob.nu = d.nu;
    prob.ny = d.ny;
    prob.nrow = d.ncstr;
    prob.A_dense = nsp.s_c_stacked;
    prob.B_dense = nsp.s_c_stacked_0_K;
    prob.a_dense = nsp.s_c_stacked_0_k;
    prob.rows = rows;
    prob.cols = cols;
}

} // namespace moto::solver::projection
