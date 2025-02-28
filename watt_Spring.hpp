#ifndef WATT_SPRING_H_
#define WATT_SPRING_H_

#include <Eigen/Dense>

namespace Watt::Spring
{
// Distinction because of TinyAD integration
template <typename DoF, typename Parameter>
DoF computeEnergy(const Eigen::Vector3<DoF> &xA, const Eigen::Vector3<DoF> &xB, Parameter k, Parameter L0)
{
    DoF L = (xA - xB).norm();
    return 0.5 * k * (L - L0) * (L - L0);
}

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
