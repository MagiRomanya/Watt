#ifndef WATT_STABLENEOHOOKEAN_H_
#define WATT_STABLENEOHOOKEAN_H_

#include <Eigen/Dense>

namespace Watt::StableNeoHookean
{

/**
 * @brief Computes the skew-symmetric matrix of a vector.
 *
 * @param v A 3D vector.
 * @return 3x3 skew-symmetric matrix of vector v.
 */
template <typename Scalar>
Eigen::Matrix3<Scalar> skew(const Eigen::Vector3<Scalar> &v)
{
    return Eigen::Matrix3<Scalar>{{0.0, -v.z(), v.y()},  //
                                  {v.z(), 0.0, -v.x()},
                                  {-v.y(), v.x(), 0.0}};
}

/**
 * @brief Computes the Stable Neo-Hookean energy density for a given deformation gradient.
 *
 *
 * Energy taken from "Dynamic Defromables" SIGG22 course by T.Kim & D.Eberle.
 * The energy is calculated as:
 *
 *     E = 0.5 * mu * (I2 - 3) - mu * (I3 - 1) + 0.5 * lambda * (I3 - 1)²
 *
 * where:
 *   - mu & lambda are the Lamé parameters
 *   - I2 = F² is the second invariant
 *   - I3 = det(F) is the third invariant
 *
 * @param F Deformation gradient (3x3 matrix).
 * @param mu Shear modulus (first Lamé parameter).
 * @param lambda Second Lamé parameter.
 * @return Strain energy density (scalar).
 */
template <typename DoF, typename Parameter>
DoF computeEnergy(const Eigen::Matrix3<DoF> &F, Parameter mu, Parameter lambda)
{
    const DoF I2 = F.squaredNorm();
    const DoF I3 = F.determinant();

    return 0.5 * mu * (I2 - 3.0) - mu * (I3 - 1.0) + 0.5 * lambda * (I3 - 1.0) * (I3 - 1.0);
}

/**
 * @brief Computes the first Piola-Kirchhoff stress tensor (PK1) for the Stable Neo-Hookean model.
 *
 * @param F Deformation gradient (3x3 matrix).
 * @param mu Shear modulus (first Lamé parameter).
 * @param lambda Second Lamé parameter.
 * @return PK1 stress tensor (3x3 matrix).
 */
template <typename Scalar>
Eigen::Matrix3<Scalar> computePK1(const Eigen::Matrix3<Scalar> &F, Scalar mu, Scalar lambda)
{
    const Scalar I3 = F.determinant();
    Eigen::Matrix3<Scalar> jacI3;
    jacI3 << F.col(1).cross(F.col(2)), F.col(2).cross(F.col(0)), F.col(0).cross(F.col(1));

    return mu * F - mu * jacI3 + lambda * (I3 - 1.0) * jacI3;
}

/**
 * @brief Computes the gradient of the energy with respect to the deformation gradient.
 *
 * The gradient is equivalent to the PK1 stress tensor, flattened as a 9D vector.
 *
 * @param F Deformation gradient (3x3 matrix).
 * @param mu Shear modulus (first Lamé parameter).
 * @param lambda Second Lamé parameter.
 * @return Gradient of the energy (vector of size 9).
 */
template <typename Scalar>
Eigen::Vector<Scalar, 9> computeEnergyGradient(const Eigen::Matrix3<Scalar> &F, Scalar mu, Scalar lambda)
{
    return computePK1(F, mu, lambda).reshaped();  // Flatten PK1
}

/**
 * @brief Computes the Hessian (second derivative) of the energy with respect to the deformation gradient.
 *
 * @param F Deformation gradient (3x3 matrix).
 * @param mu Shear modulus (first Lamé parameter).
 * @param lambda Second Lamé parameter.
 * @return Hessian matrix (9x9).
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 9, 9> computeEnergyHessian(const Eigen::Matrix3<Scalar> &F, Scalar mu, Scalar lambda)
{
    using Mat9 = Eigen::Matrix<Scalar, 9, 9>;

    const Scalar I3 = F.determinant();

    Eigen::Matrix3<Scalar> jacI3;
    jacI3 << F.col(1).cross(F.col(2)), F.col(2).cross(F.col(0)), F.col(0).cross(F.col(1));

    Mat9 hessI3 = Mat9::Zero();
    hessI3.template block<3, 3>(0, 3) = -skew<Scalar>(F.col(2));
    hessI3.template block<3, 3>(3, 0) = -hessI3.template block<3, 3>(0, 3);

    hessI3.template block<3, 3>(0, 6) = skew<Scalar>(F.col(1));
    hessI3.template block<3, 3>(6, 0) = -hessI3.template block<3, 3>(0, 6);

    hessI3.template block<3, 3>(3, 6) = -skew<Scalar>(F.col(0));
    hessI3.template block<3, 3>(6, 3) = -hessI3.template block<3, 3>(3, 6);

    return mu * Mat9::Identity() - mu * hessI3 + lambda * (I3 - 1.0) * hessI3 +
           lambda * jacI3.reshaped() * jacI3.reshaped().transpose();
}

};  // namespace Watt::StableNeoHookean

#endif  // WATT_STABLENEOHOOKEAN_H_
