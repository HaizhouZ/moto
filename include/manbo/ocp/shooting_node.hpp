#ifndef __SHOOTING_NODE__
#define __SHOOTING_NODE__

#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include <stack>
#include <array>

#include <manbo/ocp/problem_formulation.hpp>
#include <manbo/core/approximation.hpp>

namespace manbo {
/**
 * @brief for loop shortcut for funcs (dyn,cost,constr...)
 *
 * @param problem
 * @param callback [idx of field, idx of expr, pointer to expr]
 */
inline void for_loop_funcs(problem_ptr_t problem,
                           std::function<void(size_t, size_t, multivariate_ptr_t)> callback) {
    for (size_t i = 0, field = field_type::dyn; field != field::num; i++, field++) {
        auto& _exprs = problem->expr_[field];
        if (_exprs.empty()) {
            for (size_t j = 0; j < _exprs.size(); j++) {
                auto _c = std::static_pointer_cast<multivariate>(_exprs[j]);
                callback(i, j, _c);
            }
        }
    }
}
class mem_mgr {
   private:
    template <typename data_type>
    struct lck_data {
        std::mutex mtx_;
        std::stack<data_type> data_;
    };

   public:
    typedef std::array<std::vector<approx_data_ptr_t>, field::num_func> stacked_approx_ptr;

    template <typename class_type>
    void add_problem(problem_ptr_t problem) {
        problems_[problem->uid_] = problem;
        approx_[problem->uid_] = lck_data<approx_data>();
    }

    static auto make_approx_data(problem_ptr_t problem) {
        stacked_approx_ptr d;
        for_loop_funcs(problem, [&d](size_t i, size_t j, multivariate_ptr_t _c) {
            if (_c->approx_level() == approx_type::first) {
                auto c = std::static_pointer_cast<first_approx>(_c);
                d[i].push_back(c->make_approx_data());
            }
        });
        return d;
    }

    void create_data_batch(problem_ptr_t problem, size_t N) {
        auto& _approx = approx_[problem->uid_];
        std::lock_guard _lock(_approx.mtx_);
        for (size_t i = 0; i < N; i++)
            _approx.data_.push(make_approx_data(problem));
    }

    stacked_approx_ptr acquire_approx(problem_ptr_t problem) {
        auto& _approx = approx_[problem->uid_];
        std::lock_guard _lock(_approx.mtx_);
        if (!_approx.data_.empty()) {
            auto p = std::move(_approx.data_.top());
            _approx.data_.pop();
            return p;
        } else {
            return make_approx_data(problem);
        }
    }

    void release_approx(problem_ptr_t problem, stacked_approx_ptr&& data) {
        auto& _approx = approx_[problem->uid_];
        std::lock_guard _lock(_approx.mtx_);
        _approx.data_.push(std::move(data));
    }

   private:
    std::map<size_t, problem_ptr_t> problems_;
    std::map<size_t, lck_data<stacked_approx_ptr>> approx_;
};
def_ptr(mem_mgr);

/**
 * @brief shooting node in an OCP
 * @todo data collection/serialization/deserialization should be finished in this node!
 */
class shooting_node {
   public:
    shooting_node(problem_ptr_t formulation, mem_mgr_ptr_t mem)
        : problem_(formulation), mem_(mem) {
        approx_ = std::move(mem_->acquire_approx(problem_));
    }

    ~shooting_node() {
        mem_->release_approx(problem_, std::move(approx_));
    }

    void swap(shooting_node& p) {
        problem_.swap(p.problem_);
        mem_.swap(p.mem_);
        approx_.swap(p.approx_);
        primal_data_.swap(p.primal_data_);
    }

    void collect_data() {
        auto collect_callback = [this](size_t field, size_t j, multivariate_ptr_t _c) {
            if (_c->approx_level() == approx_type::first) {
                auto c = std::static_pointer_cast<first_approx>(_c);
                c->evaluate(problem_, primal_data_, approx_[field][j]);
            }
        };
        for_loop_funcs(problem_, collect_callback);
    }

   private:
    problem_ptr_t problem_;
    mem_mgr_ptr_t mem_;
    mem_mgr::stacked_approx_ptr approx_;
    primal_data_ptr_t primal_data_;
};
}  // namespace manbo

#endif /*__NODE_*/