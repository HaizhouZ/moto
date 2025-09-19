#include <moto/ocp/impl/custom_func.hpp>
#include <moto/ocp/impl/func_data.hpp>
#include <moto/ocp/problem.hpp>

namespace moto {
shared_data::shared_data(ocp *prob, sym_data *primal) : prob_(prob) {
    prob->wait_until_ready();
    data_.reserve(moto::custom_func_fields.size());
    for (auto field : moto::custom_func_fields) {
        for (const generic_custom_func &f : prob->exprs(field)) {
            if (data_.find(f.uid()) == data_.end()) {
                add(f.uid(), f.create_custom_data(*primal, *this));
            }
        }
    }
}
} // namespace moto