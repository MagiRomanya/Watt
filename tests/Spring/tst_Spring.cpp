#include <catch2/catch_all.hpp>
#include <TinyAD/Scalar.hh>

#include "watt_Spring.hpp"

TEST_CASE("SPRING ENERGY DERIVATIVES")
{
    using StateVec = Eigen::Vector<double, 6>;
    StateVec x = StateVec::Zero();

    constexpr auto size = StateVec::SizeAtCompileTime;
    constexpr auto nTests = 1024;
    constexpr auto tol = 1e-8;

    auto deformation = Catch::Generators::generate("displacement generator", CATCH_INTERNAL_LINEINFO, [nTests] {
        // NOLINTNEXTLINE
        using namespace Catch::Generators;
        return makeGenerators(
            take(nTests, map([](const auto &) { return (1.0 * StateVec::Random()).eval(); }, range(0, nTests))));
    });

    x += deformation;

    constexpr double L0 = 1.0;
    constexpr double k = 1.0;

    using ADouble = TinyAD::Scalar<size, double>;
    Eigen::Vector<ADouble, 6> xAD = ADouble::make_active(x);

    const ADouble energyAD = Watt::Spring::computeEnergy<ADouble>(xAD.segment<3>(0), xAD.segment<3>(3), k, L0);
    SECTION("Energy")
    {
        const double energyWatt = Watt::Spring::computeEnergy<double>(x.segment<3>(0), x.segment<3>(3), k, L0);
        REQUIRE_THAT(energyWatt - energyAD.val, Catch::Matchers::WithinAbs(0, tol));
    }
    SECTION("Energy Gradient")
    {
        const Eigen::Vector<double, 6> gradientWatt =
            Watt::Spring::computeEnergyGradient<double>(x.segment<3>(0), x.segment<3>(3), k, L0);
        REQUIRE_THAT((gradientWatt - energyAD.grad).norm(), Catch::Matchers::WithinAbs(0, tol));
    }

    SECTION("Energy Hessian")
    {
        const Eigen::Matrix<double, 6, 6> hessianWatt =
            Watt::Spring::computeEnergyHessian<double>(x.segment<3>(0), x.segment<3>(3), k, L0);

        REQUIRE_THAT((hessianWatt - energyAD.Hess).norm(), Catch::Matchers::WithinAbs(0, tol));
    }
}
