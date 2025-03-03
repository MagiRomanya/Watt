#ifndef WATT_SPRING_H_
#define WATT_SPRING_H_

#include <Eigen/Dense>

namespace Watt::Spring
{
/**
 * @brief Computes the elastic potential energy of a linear spring between two points.
 *
 * The energy is calculated as:
 *
 *     E = 0.5 * k * (L - L0)²
 *
 * where:
 *   - k is the spring stiffness.
 *   - L is the current length of the spring.
 *   - L0 is the rest (initial) length of the spring.
 *
 * @tparam DoF Degree of Freedom type (e.g., double, TinyAD::Dual).
 * @tparam Parameter Parameter type (typically double).
 * @param xA Position of the first endpoint of the spring.
 * @param xB Position of the second endpoint of the spring.
 * @param k Spring stiffness coefficient.
 * @param L0 Rest length of the spring.
 * @return The scalar elastic potential energy of the spring.
 */
template <typename DoF, typename Parameter>
DoF computeEnergy(const Eigen::Vector3<DoF> &xA, const Eigen::Vector3<DoF> &xB, Parameter k, Parameter L0)
{
    DoF L = (xA - xB).norm();
    return 0.5 * k * (L - L0) * (L - L0);
}

/**
 * @brief Computes the gradient of the spring energy with respect to the node positions.
 *
 * Layout of the gradient:
 *   - First 3 entries: gradient with respect to xA.
 *   - Last 3 entries: gradient with respect to xB.
 *
 * @tparam Scalar Numeric type (e.g., float, double).
 * @param xA Position of the first endpoint of the spring.
 * @param xB Position of the second endpoint of the spring.
 * @param k Spring stiffness coefficient.
 * @param L0 Rest length of the spring.
 * @return Eigen::Vector<Scalar, 6> The gradient of the energy
 */
template <typename Scalar>
Eigen::Vector<Scalar, 6> computeEnergyGradient(const Eigen::Vector3<Scalar> &xA,
                                               const Eigen::Vector3<Scalar> &xB,
                                               Scalar k,
                                               Scalar L0)
{
    Eigen::Vector<Scalar, 6> gradient;
    Scalar L = (xA - xB).norm();
    gradient.template segment<3>(0) = k * (L - L0) * (xA - xB) / L;
    gradient.template segment<3>(3) = -gradient.template segment<3>(0);
    return gradient;
}

/**
 * @brief Computes the Hessian matrix of the spring energy
 *        with respect to the node positions.
 *
 * @tparam Scalar Numeric type (e.g., float, double).
 * @param xA Position of the first endpoint of the spring.
 * @param xB Position of the second endpoint of the spring.
 * @param k Spring stiffness coefficient.
 * @param L0 Rest length of the spring.
 * @return Eigen::Matrix<Scalar, 6, 6> The Hessian matrix of the energy.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 6, 6> computeEnergyHessian(const Eigen::Vector3<Scalar> &xA,
                                                 const Eigen::Vector3<Scalar> &xB,
                                                 Scalar k,
                                                 Scalar L0)
{
    Eigen::Matrix<Scalar, 6, 6> hessian;
    Scalar L = (xA - xB).norm();
    const Eigen::Vector3<Scalar> u = (xA - xB) / L;
    hessian.template block<3, 3>(0, 0) =
        k / L * ((L - L0) * Eigen::Matrix3<Scalar>::Identity() + L0 * u * u.transpose());
    hessian.template block<3, 3>(0, 3) = -hessian.template block<3, 3>(0, 0);
    hessian.template block<3, 3>(3, 0) = -hessian.template block<3, 3>(0, 0);
    hessian.template block<3, 3>(3, 3) = hessian.template block<3, 3>(0, 0);
    return hessian;
}

};  // namespace Watt::Spring

#endif  // WATT_SPRING_H_
