
#ifndef MOTO_UTILS_MOVABLE_PTR_HPP
#define MOTO_UTILS_MOVABLE_PTR_HPP

template <typename T>
class movable_ptr {
  private:
    T *ptr = nullptr;

  public:
    movable_ptr() = default;
    movable_ptr(T *p) : ptr(p) {}
    movable_ptr(movable_ptr &&rhs) : ptr(rhs.ptr) { rhs.ptr = nullptr; }
    movable_ptr(const movable_ptr &) = delete;
    movable_ptr &operator=(const movable_ptr &) = delete;
    T *operator->() { return ptr; }
    T &operator*() { return *ptr; }
    T *get() { return ptr; }
    bool operator==(const movable_ptr &rhs) const { return ptr == rhs.ptr; }
    operator bool() const { return ptr != nullptr; }
};

#endif