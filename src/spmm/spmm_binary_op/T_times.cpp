#include <moto/spmm/impl/spmm_binary_op.hpp>

#include <moto/spmm/impl/binary_op_helper.hpp>

namespace moto {

template <bool add, typename rhs_type, typename out_type>
void sparse_mat::T_times(const rhs_type &rhs, out_type &out) {
    assert((spmm::is_consistent<rhs_type, out_type>::value(spmm::T_times)) && "inconsistent dimensions for T_times");
    assert(rhs.rows() == rows_ && "rhs matrix size mismatch");
    assert(rhs.cols() == out.cols() && "rhs matrix size mismatch");
    assert(cols_ == out.rows() && "out matrix size mismatch");
    spmm::product<true, false, add>(*this, rhs, out);
}
EXPLICIT_SP_MEMFUNC_INSTANTIATE(T_times)

}