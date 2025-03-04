#include <catch2/catch_all.hpp>
#include <unsupported/Eigen/KroneckerProduct>

#include <tuple>

#include "watt_KroneckerProduct.hpp"

template <typename MatA, typename MatB>
void testKroneckerProduct()
{
    MatA A = MatA::Random();
    MatB B = MatB::Random();

    auto AB_Watt = Watt::KroneckerProduct::kroneckerProduct(A, B).eval();
    auto AB_Eigen = Eigen::KroneckerProduct(A, B).eval();
    REQUIRE_THAT((AB_Watt - AB_Eigen).norm(), Catch::Matchers::WithinAbs(0, 1e-8));
}

// Helper to iterate over tuple pairs
template <typename Tuple, std::size_t I = 0, std::size_t J = 0>
constexpr void for_each_pair(Tuple types)
{
    if constexpr (I < std::tuple_size_v<Tuple>) {
        if constexpr (J < std::tuple_size_v<Tuple>) {
            using MatA = std::tuple_element_t<I, Tuple>;
            using MatB = std::tuple_element_t<J, Tuple>;
            testKroneckerProduct<MatA, MatB>();
            for_each_pair<Tuple, I, J + 1>(types);
        } else {
            for_each_pair<Tuple, I + 1, 0>(types);
        }
    }
}

TEST_CASE("Kronecker Product")
{
    // Types
    auto matrixTypes = std::tuple<  //
        Eigen::Matrix<double, 1, 2>,
        Eigen::Matrix<double, 2, 1>,
        Eigen::Matrix<double, 2, 2>,
        Eigen::Matrix<double, 2, 3>,
        Eigen::Matrix<double, 3, 2>,
        Eigen::Matrix<double, 3, 3>,
        Eigen::Matrix<double, 4, 3>,
        Eigen::Matrix<double, 3, 4>,
        Eigen::Matrix<double, 4, 4>,
        Eigen::Matrix<double, 5, 4>,
        Eigen::Matrix<double, 4, 5>,
        Eigen::Matrix<double, 5, 5>>{};

    constexpr auto nTests = 1024;

    for (std::size_t i = 0; i < nTests; ++i) {
        for_each_pair(matrixTypes);
    }
}
