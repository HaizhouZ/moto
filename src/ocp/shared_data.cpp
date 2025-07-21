#include <moto/ocp/impl/func_data.hpp>
#include <moto/ocp/impl/custom_func.hpp>

namespace moto {
shared_data::shared_data(const ocp_ptr_t &prob, sym_data &primal) : prob_(prob) {
    data_.reserve(moto::custom_func_fields.size());
    for (auto f : moto::custom_func_fields) {
        for (const auto &expr : prob->exprs(f)) {
            if (data_.find(expr->uid()) == data_.end()) {
                add(expr->uid(), static_cast<const custom_func *>(expr)->create_custom_data(primal, *this));
            }
        }
    }
}
} // namespace moto