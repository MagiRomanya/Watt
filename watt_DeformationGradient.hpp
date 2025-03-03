#ifndef WATT_DEFORMATIONGRADIENT_H_
#define WATT_DEFORMATIONGRADIENT_H_

#include <Eigen/Dense>

namespace Watt::DeformationGradient
{

/**
 * @brief Computes the edge matrix of a tetrahedron in 3D space.
 *
 * The edge matrix D is constructed from the edges formed between the first vertex (xA)
 * and the other three vertices (xB, xC, xD) of the tetrahedron:
 *
 *     D = [ xB - xA | xC - xA | xD - xA ]
 *
 * @tparam Scalar Numeric type (e.g., float, double).
 * @param xA First vertex of the tetrahedron (serves as the origin).
 * @param xB Second vertex of the tetrahedron.
 * @param xC Third vertex of the tetrahedron.
 * @param xD Fourth vertex of the tetrahedron.
 * @return Eigen::Matrix3<Scalar> The 3x3 edge matrix D.
 */
template <typename Scalar>
Eigen::Matrix3<Scalar> computeEdgeMatrix3D(const Eigen::Vector3<Scalar> &xA,
                                           const Eigen::Vector3<Scalar> &xB,
                                           const Eigen::Vector3<Scalar> &xC,
                                           const Eigen::Vector3<Scalar> &xD)
{
    Eigen::Matrix3<Scalar> D;
    D.col(0) = xB - xA;
    D.col(1) = xC - xA;
    D.col(2) = xD - xA;
    return D;
}

/**
 * @brief Computes the deformation gradient F of a tetrahedron.
 *
 * The deformation gradient F maps from the rest configuration to the deformed configuration:
 *
 *     F = Ds * invDm
 *
 * where:
 *   - Ds is the edge matrix in the deformed (world) configuration.
 *   - invDm is the inverse of the edge matrix in the rest configuration.
 *
 * @tparam Scalar Numeric type (e.g., float, double).
 * @param Ds Edge matrix of the deformed tetrahedron.
 * @param invDm Inverse of the edge matrix of the rest (reference) tetrahedron.
 * @return Eigen::Matrix3<Scalar> The 3x3 deformation gradient F.
 */
template <typename Scalar>
Eigen::Matrix3<Scalar> computeDeformationGradient(const Eigen::Matrix3<Scalar> &Ds, const Eigen::Matrix3<Scalar> &invDm)
{
    return Ds * invDm;
}

/**
 * @brief Computes the Jacobian of the deformation gradient F with respect to the world-space positions of the
 * tetrahedron's vertices.
 *
 * The Jacobian dF/dx is a 9x12 matrix where:
 *   - 9 rows correspond to the flattened deformation gradient F (3x3 matrix).
 *   - 12 columns correspond to the derivatives with respect to the positions of the 4 vertices (each with 3
 * coordinates).
 *
 * @tparam Scalar Numeric type (e.g., float, double).
 * @param invDm Inverse of the edge matrix of the rest configuration.
 * @return Eigen::Matrix<Scalar, 9, 12> The Jacobian matrix dF/dx.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 9, 12> computeDeformationGradientJacobian(const Eigen::Matrix3<Scalar> &invDm)
{
    Eigen::Matrix<Scalar, 9, 12> dFdx;

    Eigen::Matrix<Scalar, 4, 3> G;
    G << -1, -1, -1,  //
        1, 0, 0,      //
        0, 1, 0,      //
        0, 0, 1;      //

    const Eigen::Matrix<Scalar, 3, 4> M = (G * invDm).transpose();
    // Kronecker Product Between M and eye(3)
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 4; ++j) {
            dFdx.template block<3, 3>(3 * i, 3 * j) = M(i, j) * Eigen::Matrix3<Scalar>::Identity();
        }
    }
    return dFdx;
}

/**
 * @brief Computes the signed volume of a tetrahedron.
 *
 * The volume is computed from the determinant of the rest edge matrix Dm:
 *
 *     volume = det(Dm) / 6
 *
 * A negative volume indicates an inverted tetrahedron (e.g., flipped orientation).
 *
 * @tparam Scalar Numeric type (e.g., float, double).
 * @param Dm Edge matrix of the tetrahedron in the rest configuration.
 * @return Scalar The signed volume of the tetrahedron.
 */
template <typename Scalar>
Scalar computeTetrahedronVolume(const Eigen::Matrix3<Scalar> &Dm)
{
    return Dm.determinant() / 6.0;
}

};  // namespace Watt::DeformationGradient

#endif  // WATT_DEFORMATIONGRADIENT_H_
