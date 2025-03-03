add_executable(tst_StableNeoHookean tst_StableNeoHookean.cpp)
target_link_libraries(tst_StableNeoHookean PUBLIC Watt Catch2::Catch2WithMain)

# Register the test
add_test(NAME tst_StableNeoHookean COMMAND tst_StableNeoHookean)
