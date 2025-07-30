
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
    movable_ptr &operator=(T *p) {
        ptr = p;
        return *this;
    }
    T *operator->() { return ptr; }
    template <typename U = T>
    typename std::enable_if<!std::is_void<U>::value, U &>::type operator*() { return *ptr; }
    template <typename U = T>
    operator const U *() const {
        return ptr;
    } ///< allow implicit conversion to T*
    template <typename U = T>
    operator U *() { return ptr; } ///< allow implicit conversion to T*
    T *get() { return ptr; }
    const T *get() const { return ptr; }
    bool operator==(const movable_ptr &rhs) const { return ptr == rhs.ptr; }
    operator bool() const { return ptr != nullptr; }
    operator T *() { return ptr; }             ///< allow implicit conversion to T*
    operator const T *() const { return ptr; } ///< allow implicit conversion to T*
};

#endif