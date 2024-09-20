#ifndef __EXPRESSION_BASE__
#define __EXPRESSION_BASE__

#include <manbo/common/fwd.hpp>

namespace manbo {
enum class approxType {
    zero = 0,
    first,
    second
};
enum class sparsity {
    dense,
    diag,
    single,
};
struct sparsity_info {
    std::vector<sparsity> type_;
    std::vector<double[4]> pos_;
};

struct exprBase {
   protected:
    approxType approxType_ = approxType::zero;
    const size_t dim_;
    const std::string& name_;

   public:
    inline approxType type() {
        return approxType_;
    }
    inline const std::string& name() {
        return name_;
    }
    inline size_t dim() {
        return dim_;
    }
    exprBase(const std::string& name, size_t dim)
        : name_(name), dim_(dim) {}
};
struct firstApprox : public exprBase {
    firstApprox(const std::string& name, size_t dim)
        : exprBase(name, dim) { approxType_ = approxType::first; }

    /**
     * @brief compute jacobian of the approximation w.r.t. x
     *
     * @param x stacked variables
     * @param jac stacked jacobian
     */
    virtual void compute_jacobian(Eigen::Ref<vector> x, Eigen::Ref<matrix> jac) {
    }
};
struct secondApprox : public firstApprox {
    secondApprox(const std::string& name, size_t dim)
        : firstApprox(name, dim) { approxType_ = approxType::second; }
    virtual void compute_hessian(Eigen::Ref<vector> x) {
    }
};
//////////////////////
struct exprData {
    
};

}  // namespace manbo

#endif /*__EXPRESSION_BASE_*/