#ifndef WATT_ROTATIONVARIANTPOLARSVD_H_
#define WATT_ROTATIONVARIANTPOLARSVD_H_

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace Watt::RotationVariantPolarAndSVD
{

/**
 * @brief Computes the Singular Value Decomposition of a matrix,
 *        adjusting for potential reflections in the U and V matrices.
 *
 * Given a matrix F, this function computes its full SVD decomposition:
 *     F = U * Sigma * V.transpose()
 * and ensures that the determinants of U and V correspond to proper rotations,
 * correcting any reflection (negative determinant) by flipping the sign of the
 * last column and the corresponding singular value.
 *
 * @tparam Derived Eigen matrix type derived from Eigen::MatrixBase.
 * @param F The input matrix to decompose.
 * @return A tuple containing:
 *         - U: Left singular vectors (rotation matrix).
 *         - Sigma: Singular values (vector).
 *         - V: Right singular vectors (rotation matrix).
 */
template <typename Derived>
auto svd_rv(const Eigen::MatrixBase<Derived> &F)
{
    using Scalar = typename Derived::Scalar;
    constexpr int Rows = Derived::RowsAtCompileTime;
    constexpr int Cols = Derived::ColsAtCompileTime;
    constexpr int MinDim = (Rows < Cols) ? Rows : Cols;

    using MatrixType = typename Derived::PlainObject;
    using VectorType = Eigen::Vector<Scalar, MinDim>;

    Eigen::JacobiSVD<MatrixType, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(
        F.derived(), Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto U = svd.matrixU();
    auto sigma = svd.singularValues();
    auto V = svd.matrixV();

    const auto detU = U.determinant();
    const auto detV = V.determinant();

    if (detU < 0 && detV > 0) {
        U.col(U.cols() - 1) *= -1;
        sigma[sigma.size() - 1] *= -1;
    } else if (detU > 0 && detV < 0) {
        V.col(V.cols() - 1) *= -1;
        sigma[sigma.size() - 1] *= -1;
    }

    return std::tuple<MatrixType, VectorType, MatrixType>{U, sigma, V};
}

/**
 * @brief Computes the rotation-variant polar decomposition of a matrix using SVD.
 *
 * The polar decomposition factors a matrix F into:
 *     F = R * S
 * where:
 *     - R is an orthogonal matrix (rotation).
 *     - S is a symmetric positive semi-definite matrix (stretch/shear).
 *
 * This implementation uses the Rotation Variant SVD (`svd_rv`) to first decompose F and
 * ensures that R is a proper rotation matrix without reflection.
 *
 * @tparam Derived Eigen matrix type derived from Eigen::MatrixBase.
 * @param F The input matrix to decompose.
 * @return A tuple containing:
 *         - R: The rotation (orthogonal) part of the decomposition.
 *         - S: The symmetric positive semi-definite stretch/shear matrix.
 */
template <typename Derived>
std::tuple<typename Derived::PlainObject, typename Derived::PlainObject> polar_rv(const Eigen::MatrixBase<Derived> &F)
{
    const auto [U, s, V] = svd_rv(F);
    const auto R = U * V.transpose();
    const auto S = R.transpose() * F;

    return {R, S};
}

}  // namespace Watt::RotationVariantPolarAndSVD

#endif  // WATT_ROTATIONVARIANTPOLARSVD_H_
