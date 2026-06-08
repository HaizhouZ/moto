#pragma once

#include <algorithm>
#include <array>
#include <moto/core/expr.hpp>
#include <unordered_map>
#include <utility>

namespace moto {

template <typename List>
class field_layout_store {
  protected:
    std::array<List, field::num> field_entries_;
    std::unordered_map<size_t, size_t> flatten_idx_, flatten_tidx_, index_by_uid_;
    std::array<size_t, field::num> field_dim_{};
    std::array<size_t, field::num_prim> field_tdim_{};

    const List &field_entries(size_t f) const { return field_entries_[f]; }
    List &field_entries(size_t f) { return field_entries_[f]; }
    const auto &field_entries() const { return field_entries_; }
    size_t field_dim(size_t f) const { return field_dim_[f]; }
    size_t field_tdim(size_t f) const { return f < field::num_prim ? field_tdim_[f] : 0; }
    size_t field_entry_count(size_t f) const { return field_entries(f).size(); }
    bool field_empty(size_t f) const { return field_entries(f).empty(); }
    bool has_entry(const expr &ex) const { return index_by_uid_.contains(ex.uid()); }
    size_t entry_index(const expr &ex) const { return index_by_uid_.at(ex.uid()); }

    void clear_layout() {
        field_dim_.fill(0);
        field_tdim_.fill(0);
        flatten_idx_.clear();
        flatten_tidx_.clear();
        index_by_uid_.clear();
    }

    void clear_entries() {
        for (auto &entries : field_entries_)
            entries.clear();
        clear_layout();
    }

    void index_entry(const expr &ex, size_t f, size_t index, size_t start, size_t tangent_start = 0) {
        field_dim_[f] += ex.dim();
        index_by_uid_[ex.uid()] = index;
        flatten_idx_[ex.uid()] = start;
        if (f < field::num_prim) {
            field_tdim_[f] += ex.tdim();
            flatten_tidx_[ex.uid()] = tangent_start;
        }
    }

    template <typename Entry> void append_entry(Entry &&entry) {
        field_entries_[entry->field()].emplace_back(std::forward<Entry>(entry));
    }

    template <typename Entry> void append_indexed_entry(Entry &&entry, size_t index) {
        const auto f = entry->field();
        index_entry(*entry, f, index, field_dim(f), field_tdim(f));
        field_entries_[f].emplace_back(std::forward<Entry>(entry));
    }

    template <typename Entries> static auto find_entry(Entries &entries, const expr &ex) {
        return std::find_if(entries.begin(), entries.end(),
                            [&ex](const auto &entry) { return entry->uid() == ex.uid(); });
    }

    static typename List::value_type take_entry(List &entries, const expr &ex) {
        auto it = find_entry(entries, ex);
        if (it == entries.end())
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        auto entry = std::move(*it);
        entries.erase(it);
        return entry;
    }

    typename List::value_type take_field_entry(const expr &ex) { return take_entry(field_entries_[ex.field()], ex); }

    void rebuild_layout() {
        clear_layout();
        for (size_t f = 0; f < field::num; ++f) {
            size_t cur = 0, tcur = 0, index = 0;
            for (const expr &ex : field_entries_[f]) {
                index_entry(ex, f, index++, cur, tcur);
                cur += ex.dim();
                if (f < field::num_prim)
                    tcur += ex.tdim();
            }
        }
    }

    static size_t field_offset(const std::unordered_map<size_t, size_t> &offsets, const expr &ex) {
        auto it = offsets.find(ex.uid());
        if (it == offsets.end())
            throw std::runtime_error(fmt::format("expr {} uid {} cannot be found", ex.name(), ex.uid()));
        return it->second;
    }

    size_t field_start(const expr &ex) const { return field_offset(flatten_idx_, ex); }
    size_t field_tangent_start(const expr &ex) const { return field_offset(flatten_tidx_, ex); }
};

} // namespace moto
