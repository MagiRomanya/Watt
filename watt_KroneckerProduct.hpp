#ifndef WATT_KRONECKERPRODUCT_H_
#define WATT_KRONECKERPRODUCT_H_

#include <Eigen/Dense>

namespace Watt::KroneckerProduct
{

/**
 * @brief Compute the Kronecker product of two fixed-size matrices.
 *
 * The Kronecker product takes two matrices A (m x n) and B (p x q),
 * and returns a larger matrix of size (m*p x n*q), where each element
 * of A is multiplied by the entire matrix B.
 *
 * @tparam DerivedA Fixed-size Eigen matrix type.
 * @tparam DerivedB Fixed-size Eigen matrix type.
 * @param A Matrix of size (m x n).
 * @param B Matrix of size (p x q).
 * @return Kronecker product matrix of size (m*p x n*q).
 *
 * @note This implementation only works with fixed-size matrices
 *       (compile-time dimensions), not dynamic-sized matrices.
 */
template <typename DerivedA, typename DerivedB>
Eigen::Matrix<
    typename DerivedA::Scalar,  //
    static_cast<Eigen::Index>(DerivedA::RowsAtCompileTime) * static_cast<Eigen::Index>(DerivedB::RowsAtCompileTime),
    static_cast<Eigen::Index>(DerivedA::ColsAtCompileTime) * static_cast<Eigen::Index>(DerivedB::ColsAtCompileTime)>
kroneckerProduct(const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &B)
{
    static_assert(DerivedA::SizeAtCompileTime, "Not implemented for dynamic matrices");
    static_assert(DerivedB::SizeAtCompileTime, "Not implemented for dynamic matrices");

    constexpr auto rowsA = static_cast<Eigen::Index>(DerivedA::RowsAtCompileTime);
    constexpr auto colsA = static_cast<Eigen::Index>(DerivedA::ColsAtCompileTime);
    constexpr auto rowsB = static_cast<Eigen::Index>(DerivedB::RowsAtCompileTime);
    constexpr auto colsB = static_cast<Eigen::Index>(DerivedB::ColsAtCompileTime);

    Eigen::Matrix<typename DerivedA::Scalar, rowsA * rowsB, colsA * colsB> result;
    for (Eigen::Index i = 0; i < rowsA; ++i) {
        for (Eigen::Index j = 0; j < colsA; ++j) {
            result.template block<rowsB, colsB>(rowsB * i, colsB * j) = A(i, j) * B;
        }
    }
    return result;
}

/**
 * @brief Compute the perfect shuffle matrix of size (N² x N²).
 *
 * The perfect shuffle matrix rearranges vectorized matrices to
 * switch between row-major and column-major vectorization orderings.
 * This is particularly useful when working with Kronecker products
 * and tensor operations.
 *
 * Mathematically, the perfect shuffle matrix K satisfies:
 *     K * vec(A) = vec(Aᵗ)
 * for an (N x N) matrix A.
 *
 * @tparam Scalar Scalar type (e.g., float, double).
 * @tparam N The size of the square matrix (N x N).
 * @return Perfect shuffle matrix of size (N² x N²).
 */
template <typename Scalar, int N>
Eigen::Matrix<Scalar, N * N, N * N> computePerfectShuffleMatrix()
{
    Eigen::Matrix<Scalar, N * N, N * N> result = Eigen::Matrix<Scalar, N * N, N * N>::Zero();
    Eigen::Matrix<Scalar, N, N> eye = Eigen::Matrix<Scalar, N, N>::Identity();
    for (Eigen::Index i = 0; i < N; ++i) {
        for (Eigen::Index j = 0; j < N; ++j) {
            result += kroneckerProduct(eye.col(i) * eye.col(j).transpose(), eye.col(j) * eye.col(i).transpose());
        }
    }
    return result;
}

};  // namespace Watt::KroneckerProduct

#endif  // WATT_KRONECKERPRODUCT_H_
