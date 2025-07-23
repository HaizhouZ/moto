#include <moto/ocp/impl/func_data.hpp>
#include <moto/ocp/impl/custom_func.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
shared_data::shared_data(const ocp *prob, sym_data *primal) : prob_(prob) {
    data_.reserve(moto::custom_func_fields.size());
    for (auto field : moto::custom_func_fields) {
        for (const custom_func &f : prob->exprs(field)) {
            if (data_.find(f.uid()) == data_.end()) {
                add(f.uid(), f.create_custom_data_(*primal, *this));
            }
        }
    }
}
} // namespace moto