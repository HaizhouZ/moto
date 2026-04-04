#ifndef MOTO_SOLVER_PROJECTION_DEFAULT_STAGE_HPP
#define MOTO_SOLVER_PROJECTION_DEFAULT_STAGE_HPP

#include <moto/solver/projection/types.hpp>

namespace moto::solver::ns_riccati {
struct ns_riccati_data;
}

namespace moto::solver::projection {

enum class equality_block_kind : uint8_t {
    projected_state,
    state_input,
};

struct compiled_equality_block {
    equality_block_kind kind = equality_block_kind::state_input;
    size_t source_begin = 0;
    size_t source_count = 0;
    size_t row_begin = 0;
    std::string group;
};

struct equality_block_spec {
    equality_block_kind kind = equality_block_kind::state_input;
    size_t source_begin = 0;
    size_t source_count = 0;
    std::string group;
};

struct compiled_default_elimination_stage {
    size_t projected_state_rows = 0;
    size_t state_input_rows = 0;
    std::vector<compiled_equality_block> blocks;
    std::vector<row_layout> rows;
    std::vector<column_layout> cols;

    compiled_default_elimination_stage() = default;
    compiled_default_elimination_stage(size_t ns, size_t nc, size_t nu) {
        configure(ns, nc, nu);
    }

    void configure(size_t ns, size_t nc, size_t nu);
    void configure_partitioned(size_t ns, size_t nc, size_t nu,
                               const std::vector<size_t> &projected_state_parts,
                               const std::vector<size_t> &state_input_parts);
    void configure_blocks(size_t ns, size_t nc, size_t nu,
                          const std::vector<equality_block_spec> &block_specs);
    size_t row_count() const { return rows.size(); }

    void prepare_factorization_problem(ns_riccati::ns_riccati_data &data) const;
    void prepare_residual_problem(ns_riccati::ns_riccati_data &data) const;
    void assemble_control_jacobian(ns_riccati::ns_riccati_data &data) const;
    void assemble_state_jacobian(ns_riccati::ns_riccati_data &data) const;
    void assemble_residual(ns_riccati::ns_riccati_data &data) const;
    void compose_particular_and_nullspace(const ns_riccati::ns_riccati_data &data,
                                          const matrix &rhs,
                                          matrix &particular,
                                          matrix &basis) const;
    void compose_particular(const ns_riccati::ns_riccati_data &data,
                            const vector &rhs,
                            vector &particular) const;
    void export_problem(const ns_riccati::ns_riccati_data &data,
                        assembled_problem &prob) const;
};

} // namespace moto::solver::projection

#endif
