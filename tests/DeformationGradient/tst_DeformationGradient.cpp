#include <catch2/catch_all.hpp>
#include <TinyAD/Scalar.hh>

#include "watt_DeformationGradient.hpp"

TEST_CASE("Deformation Gradient")
{
    constexpr auto tol = 1e-8;
    constexpr auto nTests = 1024;
    SECTION("3D")
    {
        constexpr int nDof = 12;
        Eigen::Vector<double, nDof> x = Eigen::Vector<double, nDof>::Random();
        Eigen::Vector<double, nDof> X = Eigen::Vector<double, nDof>::Random();

        auto deformation = Catch::Generators::generate("displacement generator", CATCH_INTERNAL_LINEINFO, [nTests] {
            // NOLINTNEXTLINE
            using namespace Catch::Generators;
            return makeGenerators(
                take(nTests,
                     map([](const auto &) { return (1.0 * Eigen::Vector<double, nDof>::Random()).eval(); },
                         range(0, nTests))));
        });

        x += deformation;
        X += deformation;

        Eigen::Matrix3d Ds = Watt::DeformationGradient::computeEdgeMatrix3D<double>(
            x.segment<3>(0), x.segment<3>(3), x.segment<3>(6), x.segment<3>(9));

        Eigen::Matrix3d Dm = Watt::DeformationGradient::computeEdgeMatrix3D<double>(
            X.segment<3>(0), X.segment<3>(3), X.segment<3>(6), X.segment<3>(9));

        Eigen::Matrix3d invDm = Dm.inverse();

        Eigen::Matrix<double, 9, 12> dFdx =
            Watt::DeformationGradient::computeDeformationGradientJacobian3D<double>(invDm);

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

    SECTION("2D")
    {
        constexpr int nDof = 6;
        Eigen::Vector<double, nDof> x = Eigen::Vector<double, nDof>::Random();
        Eigen::Vector<double, nDof> X = Eigen::Vector<double, nDof>::Random();

        auto deformation = Catch::Generators::generate("displacement generator", CATCH_INTERNAL_LINEINFO, [nTests] {
            // NOLINTNEXTLINE
            using namespace Catch::Generators;
            return makeGenerators(
                take(nTests,
                     map([](const auto &) { return (1.0 * Eigen::Vector<double, nDof>::Random()).eval(); },
                         range(0, nTests))));
        });

        x += deformation;
        X += deformation;

        Eigen::Matrix2d Ds =
            Watt::DeformationGradient::computeEdgeMatrix2D<double>(x.segment<2>(0), x.segment<2>(2), x.segment<2>(4));

        Eigen::Matrix2d Dm =
            Watt::DeformationGradient::computeEdgeMatrix2D<double>(X.segment<2>(0), X.segment<2>(2), X.segment<2>(4));

        Eigen::Matrix2d invDm = Dm.inverse();

        Eigen::Matrix<double, 4, 6> dFdx =
            Watt::DeformationGradient::computeDeformationGradientJacobian2D<double>(invDm);

        using ADouble = TinyAD::Scalar<nDof, double, false>;
        Eigen::Vector<ADouble, nDof> xAD = ADouble::make_active(x);
        Eigen::Matrix2<ADouble> DsAD = Watt::DeformationGradient::computeEdgeMatrix2D<ADouble>(
            xAD.segment<2>(0), xAD.segment<2>(2), xAD.segment<2>(4));

        Eigen::Matrix2<double> F = Ds * invDm;
        Eigen::Matrix2<ADouble> F_AD = DsAD * invDm;

        REQUIRE_THAT((F - TinyAD::to_passive(F_AD)).norm(), Catch::Matchers::WithinAbs(0, tol));

        for (unsigned int i = 0; i < 2; ++i) {
            for (unsigned int j = 0; j < 2; ++j) {
                const Eigen::Vector<double, nDof> dF_dxi_AD = F_AD(i, j).grad;
                const Eigen::Vector<double, nDof> dF_dxi_Watt = dFdx.row(2 * j + i);
                REQUIRE_THAT((dF_dxi_AD - dF_dxi_Watt).norm(), Catch::Matchers::WithinAbs(0, tol));
            }
        }
    }
}
