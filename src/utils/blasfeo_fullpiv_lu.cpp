#include <Eigen/Dense>
#include <algorithm>
#include <blasfeo.h>
#include <cmath>
#include <limits>
#include <moto/utils/blasfeo_factorizer/blasfeo_fullpiv_lu.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace moto::utils {
namespace {
using Mat = matrix;
double absolute_maximum(const double *values, int count) {
    // No-copy reduction using Eigen's configured vectorization, including
    // NaN propagation. Old BLASFEO norm implementations call scalar fmax.
    return Eigen::Map<const Eigen::ArrayXd, Eigen::Unaligned>(values, count)
        .abs().maxCoeff<Eigen::PropagateNaN>();
}
struct Packed {
    blasfeo_dmat a{};
    void *memory = nullptr;
    int m = -1, n = -1;
    Packed() = default;
    Packed(const Packed &) = delete;
    Packed &operator=(const Packed &) = delete;
    ~Packed() {
        if (memory)
            v_free_align(memory);
    }
    void resize(int rows, int cols) {
        if (rows <= m && cols <= n) {
            a.m = rows;
            a.n = cols;
            return;
        }
        if (memory)
            v_free_align(memory);
        m = std::max({1, m, rows});
        n = std::max({1, n, cols});
        v_zeros_align(&memory, blasfeo_memsize_dmat(m, n));
        blasfeo_create_dmat(m, n, &a, memory);
        a.m = rows;
        a.n = cols;
    }
};
struct Vector {
    blasfeo_dvec a{};
    void *memory = nullptr;
    int n = -1;
    ~Vector() {
        if (memory)
            v_free_align(memory);
    }
    Vector() = default;
    Vector(const Vector &) = delete;
    void resize(int size) {
        if (size <= n) {
            a.m = size;
            return;
        }
        if (memory)
            v_free_align(memory);
        n = std::max(1, size);
        v_zeros_align(&memory, blasfeo_memsize_dvec(n));
        blasfeo_create_dvec(n, &a, memory);
        a.m = size;
    }
};
} // namespace

struct blasfeo_fullpiv_lu::impl {
    Packed factor_, rhs_, kernel_rhs_, compacted_upper_, pivot_panel_;
    Vector pivot_column_, pivot_search_, transpose_rhs_;
    Mat kernel_dense_;
    std::vector<int> rows_, cols_, significant_, free_columns_;
    int m_ = 0, n_ = 0, exact_ = 0, rank_ = 0;
    double maximum_ = 0.;
    bool computed_ = false;

    double &entry(int i, int j) { return BLASFEO_DMATEL(&factor_.a, i, j); }
    double entry(int i, int j) const { return BLASFEO_DMATEL(&factor_.a, i, j); }
    void init(int m, int n) {
        computed_ = false;
        m_ = m;
        n_ = n;
        exact_ = rank_ = 0;
        maximum_ = 0.;
        rows_.resize(m);
        cols_.resize(n);
        significant_.clear();
        std::iota(rows_.begin(), rows_.end(), 0);
        std::iota(cols_.begin(), cols_.end(), 0);
        factor_.resize(m, n);
        pivot_column_.resize(m);
        pivot_search_.resize(std::max(m, n));
    }
    void eliminate() {
        for (int k = 0; k < std::min(m_, n_); k++) {
            double biggest = 0.;
            int row = k, col = k;
            auto consider = [&](double *values, int count, int first_row, int panel_rows) {
                const double score = absolute_maximum(values, count);
                if (!std::isfinite(score))
                    throw std::runtime_error("nonfinite LU intermediate");
                if (score == 0. || score < biggest)
                    return;
                for (int index = 0; index < count; index++)
                    if (std::abs(values[index]) == score) {
                        const int r = first_row + index % panel_rows, c = k + index / panel_rows;
                        if (score > biggest || c < col || (c == col && r < row)) {
                            biggest = score;
                            row = r;
                            col = c;
                        }
                        break;
                    }
            };
#if defined(MF_PANELMAJ)
            const int width = n_ - k;
            for (int first = k / D_PS * D_PS; first < m_; first += D_PS) {
                const int begin = std::max(k, first), end = std::min(m_, first + D_PS);
                double *values;
                if (end - begin == D_PS)
                    values = &entry(first, k);
                else {
                    // Exclude eliminated prefix rows and physical tail padding
                    // without modifying the factor or scanning stale entries.
                    pivot_panel_.resize(D_PS, width);
                    blasfeo_dgese(D_PS, width, 0., &pivot_panel_.a, 0, 0);
                    blasfeo_dgecp(end - begin, width, &factor_.a, begin, k, &pivot_panel_.a,
                                  begin - first, 0);
                    values = pivot_panel_.a.pA;
                }
                consider(values, width * D_PS, first, D_PS);
            }
#else
            for (int j = k; j < n_; j++) {
                blasfeo_dcolex(m_ - k, &factor_.a, k, j, &pivot_search_.a, 0);
                const double score = absolute_maximum(pivot_search_.a.pa, m_ - k);
                if (!std::isfinite(score))
                    throw std::runtime_error("nonfinite LU intermediate");
                if (score > biggest) {
                    biggest = score;
                    col = j;
                    for (int i = 0; i < m_ - k; i++)
                        if (std::abs(pivot_search_.a.pa[i]) == score) {
                            row = k + i;
                            break;
                        }
                }
            }
#endif
            if (biggest == 0.)
                break;
            maximum_ = std::max(maximum_, biggest);
            if (row != k) {
                blasfeo_drowsw(n_, &factor_.a, k, 0, &factor_.a, row, 0);
                std::swap(rows_[k], rows_[row]);
            }
            if (col != k) {
                blasfeo_dcolsw(m_, &factor_.a, 0, k, &factor_.a, 0, col);
                std::swap(cols_[k], cols_[col]);
            }
            const double pivot = entry(k, k);
            const int height = m_ - k - 1;
            if (height) {
                // Preserve direct division: reciprocal scaling changes tiny
                // Schur factors and redundant multiplier representatives.
                for (int i = k + 1; i < m_; i++)
                    entry(i, k) /= pivot;
                blasfeo_dcolex(height, &factor_.a, k + 1, k, &pivot_column_.a, 0);
                for (int j = k + 1; j < n_; j++)
                    blasfeo_dcolad(height, -entry(k, j), &pivot_column_.a, 0, &factor_.a, k + 1, j);
            }
            ++exact_;
        }
        factor_.a.use_dA = 0;
        const double cutoff = std::numeric_limits<double>::epsilon() * std::min(m_, n_) * maximum_;
        for (int k = 0; k < exact_; k++)
            if (std::abs(entry(k, k)) > cutoff)
                significant_.push_back(k);
        rank_ = significant_.size();
        computed_ = true;
    }

  public:
    void compute(const Mat &A) {
        if (!A.allFinite())
            throw std::invalid_argument("nonfinite LU input");
        init(A.rows(), A.cols());
        // BLASFEO's pack signature is non-const; it only reads the source.
        if (A.size())
            blasfeo_pack_dmat(m_, n_, const_cast<double *>(A.data()), m_, &factor_.a, 0, 0);
        eliminate();
    }
    Mat reconstruction() const {
        int p = std::min(m_, n_);
        Mat L = Mat::Zero(m_, p), U = Mat::Zero(p, n_);
        for (int i = 0; i < m_; i++)
            for (int j = 0; j < p; j++)
                if (i == j)
                    L(i, j) = 1.;
                else if (i > j)
                    L(i, j) = entry(i, j);
        for (int i = 0; i < p; i++)
            for (int j = i; j < n_; j++)
                U(i, j) = entry(i, j);
        Mat PAQ = L * U, A(m_, n_);
        for (int i = 0; i < m_; i++)
            for (int j = 0; j < n_; j++)
                A(rows_[i], cols_[j]) = PAQ(i, j);
        return A;
    }
    void kernel(Mat &Z) {
        const int free = n_ - rank_;
        Z.setZero(n_, free);
        if (!free)
            return;
        if (!rank_) {
            Z.setIdentity(n_, n_);
            return;
        }
        bool prefix = true;
        for (int i = 0; i < rank_; i++)
            prefix = prefix && significant_[i] == i;
        kernel_rhs_.resize(rank_, free);
        if (prefix) {
            blasfeo_dgecp(rank_, free, &factor_.a, 0, rank_, &kernel_rhs_.a, 0, 0);
            blasfeo_dtrsm_lunn(rank_, free, 1., &factor_.a, 0, 0, &kernel_rhs_.a, 0, 0,
                               &kernel_rhs_.a, 0, 0);
            kernel_dense_.resize(rank_, free);
            blasfeo_unpack_dmat(rank_, free, &kernel_rhs_.a, 0, 0, kernel_dense_.data(), rank_);
        } else {
            // Choose actual U pivot columns. Packed entries below a pivot
            // belong to L and must never enter the kernel equation.
            free_columns_.clear();
            for (int j = 0, p = 0; j < n_; j++) {
                if (p < rank_ && j == significant_[p])
                    ++p;
                else
                    free_columns_.push_back(j);
            }
            compacted_upper_.resize(rank_, rank_);
            for (int i = 0; i < rank_; i++) {
                for (int j = 0; j < rank_; j++)
                    BLASFEO_DMATEL(&compacted_upper_.a, i, j) =
                        j < i ? 0. : entry(significant_[i], significant_[j]);
                for (int j = 0; j < free; j++)
                    BLASFEO_DMATEL(&kernel_rhs_.a, i, j) =
                        free_columns_[j] < significant_[i]
                            ? 0.
                            : entry(significant_[i], free_columns_[j]);
            }
            compacted_upper_.a.use_dA = 0;
            blasfeo_dtrsm_lunn(rank_, free, 1., &compacted_upper_.a, 0, 0, &kernel_rhs_.a, 0, 0,
                               &kernel_rhs_.a, 0, 0);
            kernel_dense_.resize(rank_, free);
            blasfeo_unpack_dmat(rank_, free, &kernel_rhs_.a, 0, 0, kernel_dense_.data(), rank_);
            for (int i = 0; i < rank_; i++)
                Z.row(cols_[significant_[i]]) = -kernel_dense_.row(i);
            for (int j = 0; j < free; j++)
                Z(cols_[free_columns_[j]], j) = 1.;
            return;
        }
        for (int i = 0; i < rank_; i++)
            Z.row(cols_[i]) = -kernel_dense_.row(i);
        for (int j = 0; j < free; j++)
            Z(cols_[rank_ + j], j) = 1.;
    }
    void solve(Eigen::Ref<const Mat> B, Eigen::Ref<Mat> X) {
        if (B.rows() != m_)
            throw std::invalid_argument("LU RHS size");
        if (X.rows() != n_ || X.cols() != B.cols())
            throw std::invalid_argument("LU output size");
        X.setZero();
        if (!rank_ || !B.cols())
            return;
        int nrhs = B.cols();
        rhs_.resize(rank_, nrhs);
        // Permute directly into packed storage; alternating matrix/vector
        // RHSs must not allocate a dense permutation temporary each iteration.
        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < rank_; i++)
                BLASFEO_DMATEL(&rhs_.a, i, j) = B(rows_[i], j);
        blasfeo_dtrsm_llnu(rank_, nrhs, 1., &factor_.a, 0, 0, &rhs_.a, 0, 0, &rhs_.a, 0, 0);
        blasfeo_dtrsm_lunn(rank_, nrhs, 1., &factor_.a, 0, 0, &rhs_.a, 0, 0, &rhs_.a, 0, 0);
        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < rank_; i++)
                X(cols_[i], j) = BLASFEO_DMATEL(&rhs_.a, i, j);
    }
    void transpose_solve(Eigen::Ref<const Mat> B, Eigen::Ref<Mat> X) {
        if (B.rows() != n_)
            throw std::invalid_argument("LU transpose RHS size");
        if (X.rows() != m_ || X.cols() != B.cols())
            throw std::invalid_argument("LU transpose output size");
        X.setZero();
        if (!rank_ || !B.cols())
            return;
        int p = std::min(m_, n_), nrhs = B.cols();
        transpose_rhs_.resize(p);
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < p; i++)
                transpose_rhs_.a.pa[i] = B(cols_[i], j);
            // Keep the unprocessed RHS tail for Eigen's redundant-dual
            // convention. Unit-lower transpose acts on all p coordinates.
            blasfeo_dtrsv_utn(rank_, &factor_.a, 0, 0, &transpose_rhs_.a, 0, &transpose_rhs_.a, 0);
            blasfeo_dtrsv_ltu(p, &factor_.a, 0, 0, &transpose_rhs_.a, 0, &transpose_rhs_.a, 0);
            for (int i = 0; i < p; i++)
                X(rows_[i], j) = transpose_rhs_.a.pa[i];
        }
    }
};
blasfeo_fullpiv_lu::blasfeo_fullpiv_lu() = default;
blasfeo_fullpiv_lu::~blasfeo_fullpiv_lu() = default;
blasfeo_fullpiv_lu::blasfeo_fullpiv_lu(blasfeo_fullpiv_lu &&) noexcept = default;
blasfeo_fullpiv_lu &blasfeo_fullpiv_lu::operator=(blasfeo_fullpiv_lu &&) noexcept = default;

void blasfeo_fullpiv_lu::compute(const matrix &a) {
    if (!data_)
        data_ = std::make_unique<impl>();
    data_->compute(a);
}
size_t blasfeo_fullpiv_lu::rank() const {
    if (!data_ || !data_->computed_)
        throw std::logic_error("LU has not been computed");
    return data_->rank_;
}
matrix blasfeo_fullpiv_lu::kernel() {
    matrix output;
    kernel(output);
    return output;
}
void blasfeo_fullpiv_lu::kernel(matrix &output) {
    if (!data_ || !data_->computed_)
        throw std::logic_error("LU has not been computed");
    data_->kernel(output);
}
matrix blasfeo_fullpiv_lu::reconstruction() const {
    if (!data_ || !data_->computed_)
        throw std::logic_error("LU has not been computed");
    return data_->reconstruction();
}
void blasfeo_fullpiv_lu::solve(Eigen::Ref<const matrix> rhs, Eigen::Ref<matrix> output) {
    if (!data_ || !data_->computed_)
        throw std::logic_error("LU has not been computed");
    data_->solve(rhs, output);
}
void blasfeo_fullpiv_lu::transpose_solve(Eigen::Ref<const matrix> rhs, Eigen::Ref<matrix> output) {
    if (!data_ || !data_->computed_)
        throw std::logic_error("LU has not been computed");
    data_->transpose_solve(rhs, output);
}
} // namespace moto::utils
