#ifndef __MULTI_LINKED_LIST_ITEM__
#define __MULTI_LINKED_LIST_ITEM__

#include <vector>
#include <memory>

namespace atri {
template <typename value_type>
class multi_linked_list_item {
    using value_ptr = std::shared_ptr<value_type>;
    value_ptr value_;
    std::vector<value_ptr> prev_;
    std::vector<value_ptr> next_;

    bool is_end() { return next_.empty(); }
    bool is_begin() { return prev_.empty(); }

    template <typename... Args>
    multi_linked_list_item(Args&&... args)
        : value_(std::make_shared<value_type>(std::forward<Args>(args)...)) {}

    multi_linked_list_item(value_ptr val)
        : value_(val) {}

    void append(value_ptr p) {
        next_.push_back(p);
        p->prev_.push_back(std::static_pointer_cast<value_type>(this));
    }
    void swap(value_ptr p_to_discard) {
        value_.swap(*p_to_discard);
    }
};

}  // namespace atri

#endif /*__MULTI_LINKED_LIST_ITEM_*/