#include <moto/spmm/impl/spmm_binary_op.hpp>

#include <moto/spmm/impl/binary_op_helper.hpp>

namespace moto {

template <bool add, typename rhs_type, typename out_type>
void sparse_mat::times(const rhs_type &rhs, out_type &out) {
    assert((spmm::is_consistent<rhs_type, out_type>::value(spmm::times)) && "inconsistent dimensions for times");
    assert(rhs.rows() == cols_ && "rhs matrix size mismatch");
    assert(rhs.cols() == out.cols() && "rhs matrix size mismatch");
    assert(rows_ == out.rows() && "out matrix size mismatch");
    spmm::product<false, false, add>(*this, rhs, out);
}
EXPLICIT_SP_MEMFUNC_INSTANTIATE(times)

}