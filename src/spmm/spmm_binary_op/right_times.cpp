#include <moto/spmm/impl/spmm_binary_op.hpp>

#include <moto/spmm/impl/binary_op_helper.hpp>

namespace moto {

template <bool add, typename lhs_type, typename out_type>
void sparse_mat::right_times(const lhs_type &lhs, out_type &out, clip_info info) const {
    if constexpr (std::is_same_v<lhs_type, row_vector> && std::is_same_v<out_type, vector>) {
        // row_vector * vector
        // assert(lhs.cols() == rows_ && "lhs matrix size mismatch");
        // assert(out.rows() == cols_ && "out matrix size mismatch");
        row_vector::AlignedMapType out_map(out.data(), out.rows());
        spmm::product<false, false, add>(lhs, *this, out_map, info);
    } else {
        assert((spmm::is_consistent<lhs_type, out_type>::value(spmm::right_times)) && "inconsistent dimensions for right_times");
        // assert(lhs.rows() == out.rows() && "lhs matrix size mismatch");
        // assert(lhs.cols() == rows_ && "lhs matrix size mismatch");
        // assert(out.cols() == cols_ && "out matrix size mismatch");
        spmm::product<false, false, add>(lhs, *this, out, info);
    }
}
EXPLICIT_SP_MEMFUNC_INSTANTIATE(right_times)

} // namespace moto