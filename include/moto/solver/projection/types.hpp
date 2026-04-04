#ifndef MOTO_SOLVER_PROJECTION_TYPES_HPP
#define MOTO_SOLVER_PROJECTION_TYPES_HPP

#include <moto/core/fwd.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace moto::solver::projection {

enum class row_kind : uint8_t {
    projected_state_eq,
    state_input_eq,
};

enum class column_kind : uint8_t {
    control,
};

struct row_layout {
    row_kind kind = row_kind::state_input_eq;
    uint32_t source_index = 0;
    uint32_t canonical_index = 0;
    std::string group;
};

struct column_layout {
    column_kind kind = column_kind::control;
    uint32_t source_index = 0;
    uint32_t canonical_index = 0;
    std::string group;
};

struct assembled_problem {
    size_t nx = 0;
    size_t nu = 0;
    size_t ny = 0;
    size_t nrow = 0;

    matrix A_dense;
    matrix B_dense;
    vector a_dense;

    std::vector<row_layout> rows;
    std::vector<column_layout> cols;

    void clear_numeric() {
        A_dense.resize(0, 0);
        B_dense.resize(0, 0);
        a_dense.resize(0);
        nrow = 0;
    }
};

enum class rank_case : uint8_t {
    unconstrained,
    constrained,
    fully_constrained,
    inconsistent,
};

struct analyze_info {
    rank_case rank_status = rank_case::unconstrained;
    size_t numeric_rank = 0;
    size_t nullity = 0;
    bool structurally_reusable = false;
};

struct stage_outputs {
    analyze_info info;
    bool has_factorization = false;
    bool has_reduced_step = false;
    bool has_duals = false;

    matrix U_p;
    vector u_p;
    matrix Y_p;
    vector y_p;

    matrix K_red;
    vector k_red;

    matrix T_u;
    matrix T_y;

    vector dlbd_proj;

    void clear_numeric() {
        has_factorization = false;
        has_reduced_step = false;
        has_duals = false;
        U_p.resize(0, 0);
        u_p.resize(0);
        Y_p.resize(0, 0);
        y_p.resize(0);
        K_red.resize(0, 0);
        k_red.resize(0);
        T_u.resize(0, 0);
        T_y.resize(0, 0);
        dlbd_proj.resize(0);
    }
};

} // namespace moto::solver::projection

#endif
