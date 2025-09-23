#ifndef MOTO_UTILS_CODEGEN_HPP
#define MOTO_UTILS_CODEGEN_HPP

#include <moto/core/fields.hpp> // Assuming moto::field_t is defined here
#include <moto/ocp/sym.hpp>

#include <mutex>

// 3rd-party includes
#include "nlohmann/json.hpp" // nlohmann/json
#include <casadi/casadi.hpp>
#include <re2/re2.h>

#include <moto/spmm/fwd.hpp> // moto::sparsity

namespace moto {
namespace utils {

// Namespace aliases
namespace fs = std::filesystem;
using json = nlohmann::json;

std::string compute_md5(const std::string &file_path);

namespace cs_codegen {

struct job_list {
    using job_type = std::function<void()>;
    std::vector<job_type> jobs;
    void wait_until_finished();
    void add(job_type &&w) {
        jobs.emplace_back(std::move(w));
    }
    void add(const job_list &other) {
        jobs.insert(jobs.end(), other.jobs.begin(), other.jobs.end());
    }
    void add(job_list &&other) {
        jobs.insert(jobs.end(), std::make_move_iterator(other.jobs.begin()), std::make_move_iterator(other.jobs.end()));
    }
};

struct task {
    std::string func_name;
    using in_arg_list_t = var_inarg_list;
    in_arg_list_t sx_inputs;
    cs::SX sx_output;
    bool gen_eval = true;
    bool gen_jacobian = false;
    bool gen_hessian = false;
    std::vector<cs::SX> jac_outputs; // for multiple outputs
    std::vector<std::pair<shared_expr, cs::SX>> ext_jac;
    std::vector<std::tuple<shared_expr, shared_expr, cs::SX>> ext_hess;
    std::vector<std::vector<sparsity>>* hess_sp = nullptr; // optional hessian sparsity pattern
    std::string output_dir = "gen";
    bool force_recompile = false;
    bool check_jac_ad = false; // check if jacobian is correct by comparing with ad
    bool append_value = false;
    bool append_jac = false;
    bool gn_hessian = false;         // use gauss-newton hessian if true
    bool keep_generated_src = false; // keep generated files
    std::string eval_compile_flag = "-O3 -DNDEBUG -march=native";
    std::string jac_compile_flag = "-O3 -DNDEBUG -march=native";
    std::string hess_compile_flag = "-O3 -DNDEBUG -march=native";
    std::string prefix = "";
    bool verbose = false; // verbose output

    struct noncopyable_task : std::unique_ptr<task> {
        using base = std::unique_ptr<task>;
        using base::base;
        noncopyable_task(const noncopyable_task &rhs) : base(nullptr) {
            if (static_cast<const base &>(rhs))
                throw std::runtime_error("noncopyable_task cannot be copied");
        }
        noncopyable_task(noncopyable_task &&) = default;
    } extra_task = nullptr; // extra task to run after this task

    void finalize(job_list &jobs);
};

// Public entry point to start code generation
job_list generate_and_compile(task &&_task);

// // Waits for all compilation threads to finish
// void wait_until_generated();
}; // namespace cs_codegen
} // namespace utils
} // namespace moto

#endif // MOTO_UTILS_CODEGEN_HPP