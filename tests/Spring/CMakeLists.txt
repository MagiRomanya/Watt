add_executable(tst_Spring tst_Spring.cpp)
target_link_libraries(tst_Spring PRIVATE Watt Catch2::Catch2WithMain)

# Register the test
add_test(NAME tst_Spring COMMAND tst_Spring)
