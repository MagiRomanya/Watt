#include <catch2/catch_all.hpp>
#include <TinyAD/Scalar.hh>

#include "watt_DeformationGradient.hpp"

TEST_CASE("Deformation Gradient")
{
    constexpr auto tol = 1e-8;
    constexpr int nDof = 12;
    Eigen::Vector<double, nDof> x = Eigen::Vector<double, nDof>::Random();
    Eigen::Vector<double, nDof> X = Eigen::Vector<double, nDof>::Random();

    constexpr auto nTests = 1024;
    auto deformation = Catch::Generators::generate("displacement generator", CATCH_INTERNAL_LINEINFO, [nTests] {
        // NOLINTNEXTLINE
        using namespace Catch::Generators;
        return makeGenerators(take(
            nTests,
            map([](const auto &) { return (1.0 * Eigen::Vector<double, nDof>::Random()).eval(); }, range(0, nTests))));
    });

    x += deformation;
    X += deformation;

    Eigen::Matrix3d Ds = Watt::DeformationGradient::computeEdgeMatrix3D<double>(
        x.segment<3>(0), x.segment<3>(3), x.segment<3>(6), x.segment<3>(9));

    Eigen::Matrix3d Dm = Watt::DeformationGradient::computeEdgeMatrix3D<double>(
        X.segment<3>(0), X.segment<3>(3), X.segment<3>(6), X.segment<3>(9));

    Eigen::Matrix3d invDm = Dm.inverse();

    Eigen::Matrix<double, 3, 4> edges;
    edges.col(0) = x.segment<3>(0);
    edges.col(1) = x.segment<3>(3);
    edges.col(2) = x.segment<3>(6);
    edges.col(3) = x.segment<3>(9);

    Eigen::Matrix<double, 9, 12> dFdx = Watt::DeformationGradient::computeDeformationGradientJacobian<double>(invDm);

    using ADouble = TinyAD::Scalar<nDof, double, false>;
    Eigen::Vector<ADouble, nDof> xAD = ADouble::make_active(x);
    Eigen::Matrix3<ADouble> DsAD = Watt::DeformationGradient::computeEdgeMatrix3D<ADouble>(
        xAD.segment<3>(0), xAD.segment<3>(3), xAD.segment<3>(6), xAD.segment<3>(9));

    Eigen::Matrix3<double> F = Ds * invDm;
    Eigen::Matrix3<ADouble> F_AD = DsAD * invDm;

    REQUIRE_THAT((F - TinyAD::to_passive(F_AD)).norm(), Catch::Matchers::WithinAbs(0, tol));

    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            const Eigen::Vector<double, nDof> dF_dxi_AD = F_AD(i, j).grad;
            const Eigen::Vector<double, nDof> dF_dxi_Watt = dFdx.row(3 * j + i);
            REQUIRE_THAT((dF_dxi_AD - dF_dxi_Watt).norm(), Catch::Matchers::WithinAbs(0, tol));
        }
    }
}
