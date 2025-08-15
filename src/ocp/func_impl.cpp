#include <condition_variable>
#include <fmt/ranges.h>
#include <moto/core/external_function.hpp>
#include <moto/ocp/impl/func.hpp>
#include <moto/utils/codegen.hpp>
#include <mutex>

namespace moto {
static bool impl_func_gen_delegated_ = false; ///< true if the codegen is delegated to @ref moto::func_codegen

func_approx_data_ptr_t generic_func::create_approx_data(sym_data &primal, merit_data &raw, shared_data &shared) const {
    if (field() - __dyn >= field::num_func)
        throw std::runtime_error(fmt::format("create_approx_data cannot be called for func {} type {}",
                                             name(), field::name(field())));
    if (in_args().empty())
        throw std::runtime_error(fmt::format("in args unset for func {} in field {}",
                                             name(), field::name(field())));
    return std::make_unique<func_approx_data>(primal, raw, shared, *this);
}
void generic_func::compute_approx(func_approx_data &data,
                                  bool eval_val, bool eval_jac, bool eval_hess) const {
    if (eval_val)
        value_impl(data);
    if (eval_jac)
        jacobian_impl(data);
    if (eval_hess)
        hessian_impl(data);
}

void generic_func::value_impl(func_approx_data &data) const {
    try {
        value(data);
    } catch (const std::bad_function_call &ex) {
        fmt::println("Function {} has no value implementation, please implement it or load from shared library", name());
    }
}
void generic_func::jacobian_impl(func_approx_data &data) const {
    try {
        jacobian(data);
    } catch (const std::bad_function_call &ex) {
        fmt::println("Function {} has no jacobian implementation, please implement it or load from shared library", name());
    }
}
void generic_func::hessian_impl(func_approx_data &data) const {
    try {
        hessian(data);
    } catch (const std::bad_function_call &ex) {
        fmt::println("Function {} has no hessian implementation, please implement it or load from shared library", name());
    }
}

void generic_func::load_external_impl(const std::string &path) {
    auto funcs = load_approx(name_, true, order_ >= approx_order::first, order_ >= approx_order::second);
    value = [eval = std::move(funcs[0])](func_approx_data &d) {
        eval.invoke(d.in_arg_data(), d.v_);
    };
    jacobian = [jac = std::move(funcs[1])](func_approx_data &d) {
        jac.invoke(d.in_arg_data(), d.jac_);
    };

    hessian = [hess = std::move(funcs[2])](func_approx_data &d) {
        hess.invoke(d.in_arg_data(), d.merit_hess_);
    };
}
void generic_func::substitute(const sym &arg, const sym &rhs) {
    if (!gen_.out_.is_empty()) {
        gen_.out_ = cs::SX::substitute(gen_.out_, arg, rhs);
    }
    auto nh = sym_uid_idx_.extract(arg.uid());
    nh.key() = rhs.uid();
    auto it = sym_uid_idx_.insert(std::move(nh)); // update the uid index
    assert(it.inserted && "substitute failed");
    in_args_.at(it.position->second) = rhs; // update the in_args_ to point to the new sym
    dep_.at(it.position->second) = rhs;     // update the dep_ to point to the new sym
}
void generic_func::set_from_casadi(const var_inarg_list &in_args, const cs::SX &out) {
    add_arguments(in_args);
    gen_.out_ = out;
}

/**
 * @brief Code generation helper for functions
 *
 */
struct func_codegen {
    void make_codegen_task(generic_func *f);

    static auto &get() {
        static func_codegen instance; ///< static instance of the codegen helper
        return instance;
    } ///< get the singleton instance

  private:
    std::mutex queue_mtx_;             //, terminate_mtx_;
    std::condition_variable queue_cv_; //, terminate_cv_;
    utils::cs_codegen::job_list job_buffer_;
    bool terminated_ = false; ///< flag to terminate the server thread
    std::thread server_;

    void add_job(utils::cs_codegen::job_list &&w) {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        job_buffer_.add(std::move(w));
        queue_cv_.notify_one();
    } ///< add a job to the codegen worker
    func_codegen() {
        server_ = std::thread([this]() {
            server(); ///< start the server thread
        });
        server_.detach(); ///< detach the server thread
    }
    ~func_codegen() {
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            terminated_ = true;     ///< set the termination flag
            queue_cv_.notify_one(); ///< notify the server to terminate}
        }
        // std::unique_lock<std::mutex> lock(terminate_mtx_);
        // terminate_cv_.wait(lock, [this] { return terminated_ == false; });
    } ///< destructor to clean up the server thread
    void server() {
        size_t n_threads = omp_get_max_threads();
        std::mutex thread_mtx_;
        std::condition_variable thread_cv_;
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this] { return !job_buffer_.jobs.empty() || terminated_; });
            if (terminated_) {
                terminated_ = false;
                break; ///< exit the loop if terminated
            }
            auto jobs = std::move(job_buffer_.jobs);
            job_buffer_.jobs.clear();
            lock.unlock();
            for (auto &w : jobs) {
                std::unique_lock<std::mutex> thread_lock(thread_mtx_);
                thread_cv_.wait(thread_lock, [&n_threads] { return n_threads > 0; });
                n_threads--;
                std::thread([&, w = std::move(w)]() mutable {
                    w();
                    std::lock_guard<std::mutex> thread_lock(thread_mtx_);
                    n_threads++;
                    thread_cv_.notify_one(); ///< notify the server that the job is done
                }).detach();
            }
        }
        // std::lock_guard<std::mutex> lock(terminate_mtx_);
        // terminate_cv_.notify_one();
    } ///< daemon to wait for codegen jobs
};

void generic_func::finalize_impl() {
    if (order_ != approx_order::none && dim_ == dim_tbd) {
        throw std::runtime_error(fmt::format("generic_func {} has no dimension set", name_));
    }
    if (!gen_.out_.is_empty()) {
        func_codegen::get().make_codegen_task(this);
    } else {
        set_ready_status(true); ///< set the ready status
    }
}
void func_codegen::make_codegen_task(generic_func *f) {
    utils::cs_codegen::task t;
    t.func_name = f->name_;
    t.sx_inputs = f->in_args_;
    t.sx_output = f->gen_.out_;
    t.gen_eval = f->order_ >= approx_order::zero;
    t.gen_jacobian = f->order_ >= approx_order::first;
    t.gen_hessian = f->order_ >= approx_order::second;
    t.append_value = f->field_ == __cost;
    t.append_jac = f->field_ == __cost;
    t.verbose = false;
    t.force_recompile = false;
    t.keep_generated_src = true;
    auto workers = utils::cs_codegen::generate_and_compile(std::move(t));
    decltype(workers) workers_set_ready_status;
    std::shared_ptr<std::atomic<size_t>> n_jobs = std::make_shared<std::atomic<size_t>>(workers.jobs.size());
    for (auto &w : workers.jobs) {
        if (w) {
            workers_set_ready_status.add([w = std::move(w), n_jobs, f]() mutable {
                w(); ///< execute the codegen job
                (*n_jobs)--;
                if (*n_jobs == 0) {
                    f->load_external_impl(); ///< load the generated code
                    f->set_ready_status(true);
                }
            });
        }
    }
    add_job(std::move(workers_set_ready_status));
}
} // namespace moto