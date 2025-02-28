#ifndef WATT_DEFORMATIONGRADIENT_H_
#define WATT_DEFORMATIONGRADIENT_H_

#include <Eigen/Dense>

namespace Watt::DeformationGradient
{

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

template <typename Scalar>
Eigen::Matrix3<Scalar> computeDeformationGradient(const Eigen::Matrix<Scalar, 3, 3> &Ds,
                                                  const Eigen::Matrix<Scalar, 3, 3> &invDm)
{
    return Ds * invDm;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 9, 12> computeDeformationGradientJacobian(const Eigen::Matrix<Scalar, 3, 3> &invDm)
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

};  // namespace Watt::DeformationGradient

#endif  // WATT_DEFORMATIONGRADIENT_H_
