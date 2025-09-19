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
        throw std::runtime_error(fmt::format("Function {} has no value implementation, please implement it or load from shared library", name()));
    }
}
void generic_func::jacobian_impl(func_approx_data &data) const {
    try {
        jacobian(data);
    } catch (const std::bad_function_call &ex) {
        throw std::runtime_error(fmt::format("Function {} has no jacobian implementation, please implement it or load from shared library", name()));
    }
}
void generic_func::hessian_impl(func_approx_data &data) const {
    try {
        hessian(data);
    } catch (const std::bad_function_call &ex) {
        throw std::runtime_error(fmt::format("Function {} has no hessian implementation, please implement it or load from shared library", name()));
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
    if (gen_.task_) {
        gen_.task_->sx_output = cs::SX::substitute(gen_.task_->sx_output, arg, rhs);
    }
    auto in_arg_it = std::find_if(in_args_.begin(), in_args_.end(),
                                  [&arg](const sym &s) { return s.uid() == arg.uid(); });
    if (in_arg_it == in_args_.end())
        throw std::runtime_error(fmt::format("func {} substitute failed: argument to replace not found", name_));
    auto in_arg_pos = in_arg_it - in_args_.begin();
    in_args_.at(in_arg_pos) = rhs; // update the in_args_ to point to the new sym
    dep_.at(in_arg_pos) = rhs;     // update the dep_ to point to the new sym
}

void generic_func::set_from_casadi(const var_inarg_list &in_args, const cs::SX &out) {
    add_arguments(in_args);
    if (gen_.task_)
        throw std::runtime_error(fmt::format("func {} already has an expression set", name_));
    else {
        gen_.task_ = new gen_info::task_type();
        gen_.task_->sx_output = out;
    }
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

generic_func generic_func::share(bool copy_args, const var_inarg_list &skip_copy_args) const {
    if (!wait_until_ready())
        throw std::runtime_error(fmt::format("func {} not ready, cannot share", name()));
    gen_.copy_task = false;
    generic_func f(*this);
    gen_.copy_task = true;
    f.finalized_ = true;                               // keep finalized
    f.async_ready_status_ = this->async_ready_status_; // share the async status
    if (copy_args) {
        bool skip_copy[in_args_.size()] = {false};
        for (const sym &s : skip_copy_args) {
            if (has_arg(s)) {
                skip_copy[sym_uid_idx_.at(s.uid())] = true;
            }
        }
        size_t idx = 0;
        for (sym &s : in_args_) {
            if (skip_copy[idx++])
                continue;
            auto s_new = s.clone();
            s_new->name() = s.name() + "_copy";
            if (bool(s.dual()) && has_arg(s.dual())) {
                skip_copy[sym_uid_idx_.at(s.dual()->uid())] = true;
                auto s_dual = s.dual()->clone();
                s_dual->name() = s.dual()->name() + "_copy";
                s_new->dual() = s_dual;
                s_dual->dual() = s_new;
            }
            f.substitute(s, s_new);
        }
    }
    return f;
}

void generic_func::finalize_impl() {
    if (gen_.task_ && !gen_.task_->sx_output.is_empty()) {
        // prune unused args
        auto &out = gen_.task_->sx_output;
        std::vector<size_t> unused_args;
        for (const sym &s : in_args_) {
            if (!cs::SX::depends_on(out, s)) {
                unused_args.push_back(s.uid());
            }
        }
        for (size_t uid : unused_args) {
            std::erase_if(in_args_, [uid](const sym &arg) { return arg.uid() == uid; });
            std::erase_if(dep_, [uid](const sym &arg) { return arg.uid() == uid; });
        }
    }
    for (auto &s : in_args_) {
        auto f = s->field();
        if (in_field(f, primal_fields)) {
            arg_dim_[f] += s->dim();
            arg_tdim_[f] += s->tdim();
            arg_num_[f]++;
            arg_by_field_[f].emplace_back(s);
        }
        sym_uid_idx_[s->uid()] = sym_uid_idx_.size();
    }
    if (order_ != approx_order::none && dim_ == dim_tbd && !zero_dim_) {
        throw std::runtime_error(fmt::format("generic_func {} has no dimension set", name_));
    }
    // setup default hessian sparsity as dense
    hess_sp_.assign(in_args_.size(), std::vector<sparsity>(in_args_.size(), default_hess_sp_));
    if (gen_.task_ && !gen_.task_->sx_output.is_empty()) {
        func_codegen::get().make_codegen_task(this);
    } else {
        set_ready_status(true); ///< set the ready status
    }
    finalized_ = true;
}

generic_func::gen_info::gen_info(const gen_info &rhs) {
    if (rhs.task_ && rhs.copy_task) {
        task_ = new task_type(*rhs.task_);
    }
}
generic_func::gen_info &generic_func::gen_info::operator=(const gen_info &rhs) {
    if (task_) {
        delete task_.get();
        task_ = nullptr;
    }
    if (rhs.task_ && rhs.copy_task) {
        task_ = new task_type(*rhs.task_);
    }
    return *this;
}
generic_func::gen_info::~gen_info() {
    if (task_) {
        delete task_.get();
        task_ = nullptr;
    }
}

void func_codegen::make_codegen_task(generic_func *f) {
    utils::cs_codegen::task &t = *f->gen_.task_;
    t.func_name = f->name_;
    t.sx_inputs = f->in_args_;
    t.gen_eval = f->order_ >= approx_order::zero;
    t.gen_jacobian = f->order_ >= approx_order::first;
    t.gen_hessian = f->order_ >= approx_order::second;
    t.append_value = f->field_ == __cost;
    t.append_jac = f->field_ == __cost;
    t.hess_sp = f->hess_sp_;
    t.verbose = false;
    t.force_recompile = false;
    t.keep_generated_src = true;
    // constexpr std::string_view debug_compile_flag = "-g -O0 -march=native";
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