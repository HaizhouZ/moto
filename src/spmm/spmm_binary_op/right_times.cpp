#include <moto/spmm/impl/spmm_binary_op.hpp>

#include <moto/spmm/impl/binary_op_helper.hpp>

namespace moto {

template <bool add, typename lhs_type, typename out_type>
void sparse_mat::right_times(const lhs_type &lhs, out_type &out) {
    assert((spmm::is_consistent<lhs_type, out_type>::value(spmm::right_times)) && "inconsistent dimensions for right_times");
    assert(lhs.rows() == out.rows() && "lhs matrix size mismatch");
    assert(lhs.cols() == rows_ && "lhs matrix size mismatch");
    assert(out.cols() == cols_ && "out matrix size mismatch");
    spmm::product<false, false, add>(lhs, *this, out);
}
EXPLICIT_SP_MEMFUNC_INSTANTIATE(right_times)

} // namespace moto