add_executable(tst_KroneckerProduct tst_KroneckerProduct.cpp)
target_link_libraries(tst_KroneckerProduct PUBLIC Watt Catch2::Catch2WithMain)

# Register the test
add_test(NAME tst_KroneckerProduct COMMAND tst_KroneckerProduct)
