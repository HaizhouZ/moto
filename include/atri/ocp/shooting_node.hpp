#ifndef __SHOOTING_NODE__
#define __SHOOTING_NODE__

#include <atri/ocp/node_data.hpp>
#include <mutex>
#include <stack>

namespace atri {

/**
 * @brief data management. this class controls the data access and allocation.
 * 1. each data_mgr controls a data_type
 * 2. data instances of same data_type can have different prob
 */
class data_mgr {
  private:
    struct data_pool : public std::stack<node_data_ptr_t> {
        std::mutex mtx_;
        data_pool() = default;
    };
    using make_data_func = std::function<node_data_ptr_t(const problem_ptr_t &)>;
    node_data_ptr_t get_data(const problem_ptr_t &prob);

    data_mgr() = default;
    data_mgr(data_mgr &) = delete;
    data_mgr(make_data_func maker) : maker_(maker) {}

  public:
    // singleton
    template <typename data_type>
    static data_mgr &get() {
        static_assert(std::is_base_of<node_data, data_type>::value,
                      "data_type must be derived from node_data");
        static_assert(
            std::is_constructible<data_type, const problem_ptr_t &>::value,
            "data_type must have a constructor that accepts [const problem_ptr_t&]");

        make_data_func maker = [](const problem_ptr_t &prob) {
            return std::make_unique<data_type>(prob);
        };

        static data_mgr s_(maker);

        return s_;
    }

    /**
     * @brief create a batch of data
     *
     * @param problem the set of expression to be computed
     * @param N number of data instances. can be seen as stages
     */
    void create_data_batch(const problem_ptr_t &problem, size_t N);
    // thread-safe data access
    node_data_ptr_t acquire_data(const problem_ptr_t &problem);
    void release_data(node_data_ptr_t &&data);

  private:
    make_data_func maker_;
    // mapping [uid of problem, data pool]
    std::unordered_map<size_t, data_pool> data_;
};

/**
 * @brief shooting node in an OCP
 * it will acquire data from data_mgr and release it back on destruction
 */
struct shooting_node {
    /**
     * @brief Construct a new shooting node object
     *
     * @param formulation problem formulation of this shootin gnode
     * @param mem data management, make sure the data_type is correct
     */
    shooting_node(problem_ptr_t formulation, data_mgr &mem)
        : problem_(formulation), mem_(mem) {
        data_ = mem_.acquire_data(problem_);
    }
    /**
     * @brief Construct a new shooting node sharing the same problem and data_mgr as rhs
     * @note it will not copy the data of rhs but get a new one
     * @param rhs the shooting ndoe to be copied
     */
    shooting_node(const shooting_node &rhs) : shooting_node(rhs.problem_, rhs.mem_) {}
    /**
     * @brief Construct a new shooting node object by moving
     * @note data will be moved to the new one
     * @param rhs the shooting node to be moved
     */
    shooting_node(shooting_node &&rhs)
        : problem_(std::move(rhs.problem_)), mem_(rhs.mem_), data_(std::move(rhs.data_)) {
    }

    ~shooting_node() {
        if (data_)
            mem_.release_data(std::move(data_));
    }
    // update the approximation
    void update_approximation();
    // get value of the sym variable
    auto value(const sym& sym) { return data_->sym_->get(sym); }
    /**
     * @brief get the sparse func data by pointer
     *
     * @param f
     * @return auto&
     */
    auto &data(func_impl *f) {
        return *data_->sparse_[f->field_][problem_->pos_by_uid_[f->uid_]];
    }
    /**
     * @brief get the sparse func data by pointer
     *
     * @param f
     * @return auto&
     */
    auto &data(const func_impl_ptr_t &f) {
        return data(f.get());
    }
    /**
     * @brief get the sparse func data by name
     * @todo check efficiency
     * @param name
     * @return auto&
     */
    auto &data(const std::string &name) {
        const auto &f = expr_index::get(name);
        if (f->field_ >= __dyn && f->field_ < field::num_constr + __dyn) {
            return data(static_cast<func_impl *>(f.get()));
        } else [[unlikely]] {
            throw std::runtime_error(
                fmt::format("func {} in field {} not supported for data()",
                            f->name_, magic_enum::enum_name(f->field_)));
        }
    }

    auto value(const std::string &name) {
        return data_->sym_->get(expr_index::get(name));
    }

    node_data_ptr_t data_;
    problem_ptr_t problem_;
    data_mgr &mem_;
};

def_ptr(shooting_node);
} // namespace atri

#endif /*__NODE_*/