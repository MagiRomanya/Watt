#include <catch2/catch_all.hpp>
#include <TinyAD/Scalar.hh>

#include <Eigen/SVD>

#include "watt_StableNeoHookean.hpp"
#include "watt_RotationVariantPolarSVD.hpp"

TEST_CASE("STABLE NEO-HOOKEAN ENERGY DERIVATIVES")
{
    constexpr auto tol = 1e-8;
    constexpr int nDof = 9;

    Eigen::Matrix3d F = Eigen::Matrix3d::Random();

    constexpr auto nTests = 1024;
    auto deformation = Catch::Generators::generate("displacement generator", CATCH_INTERNAL_LINEINFO, [nTests] {
        // NOLINTNEXTLINE
        using namespace Catch::Generators;
        return makeGenerators(
            take(nTests, map([](const auto &) { return (1.0 * Eigen::Matrix3d::Random()).eval(); }, range(0, nTests))));
    });

    F += deformation;

    const double mu = 1.0;
    const double lambda = 1.0;

    using ADouble = TinyAD::Double<nDof>;
    const Eigen::Vector<double, 9> vecF = F.reshaped();
    const Eigen::Vector<ADouble, 9> vecF_AD = ADouble::make_active(vecF);
    const Eigen::Matrix3<ADouble> F_AD = vecF_AD.reshaped(3, 3);

    const ADouble energyAD = Watt::StableNeoHookean::computeEnergy<ADouble>(F_AD, mu, lambda);
    SECTION("Energy")
    {
        const double energyWatt = Watt::StableNeoHookean::computeEnergy<double>(F, mu, lambda);
        REQUIRE_THAT(energyWatt - energyAD.val, Catch::Matchers::WithinAbs(0, tol));
    }

    SECTION("Energy Gradient")
    {
        const Eigen::Vector<double, 9> gradientWatt =
            Watt::StableNeoHookean::computeEnergyGradient<double>(F, mu, lambda);
        REQUIRE_THAT((gradientWatt - energyAD.grad).norm(), Catch::Matchers::WithinAbs(0, tol));
    }
    SECTION("Energy Hessian")
    {
        Eigen::Matrix<double, 9, 9> hessianWatt = Watt::StableNeoHookean::computeEnergyHessian(F, mu, lambda);
        REQUIRE_THAT((hessianWatt - energyAD.Hess).norm(), Catch::Matchers::WithinAbs(0, tol));
    }

    SECTION("Energy Hessian PSD")
    {
        // Compute Watt's implementation
        Eigen::JacobiSVD<Eigen::Matrix3<double>> svdSolver;
        const auto [U, sigma, V] = Watt::RotationVariantPolarAndSVD::svd_rv(F);
        Eigen::Matrix<double, 9, 9> hessPSD =
            Watt::StableNeoHookean::computeEnergyHessianPSD<double>(U, sigma, V.transpose(), mu, lambda);

        // Compute reference (AD + numerical eigensystem)
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver;
        solver.compute(energyAD.Hess);
        Eigen::Vector<double, 9> eigenvalues = solver.eigenvalues();
        Eigen::Matrix<double, 9, 9> eigenvectors = solver.eigenvectors();

        // Clamp eigenvalues to be >0
        eigenvalues = eigenvalues.cwiseMax(0);
        Eigen::Matrix<double, 9, 9> hessPSD_AD = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

        REQUIRE_THAT((hessPSD - hessPSD_AD).norm(), Catch::Matchers::WithinAbs(0, tol));
    }
}
