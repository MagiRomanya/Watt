add_executable(tst_DeformationGradient tst_DeformationGradient.cpp)
target_link_libraries(tst_DeformationGradient PUBLIC Watt Catch2::Catch2WithMain)

# Register the test
add_test(NAME tst_DeformationGradient COMMAND tst_DeformationGradient)
