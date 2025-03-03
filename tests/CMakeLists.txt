cmake_minimum_required(VERSION 3.10)

# Project name
project(WATT_TESTS)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(CTest)
enable_testing()

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)

FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0  
)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(Eigen)
set(BUILD_TESTING ON)

# Fetch TinyAD
FetchContent_Declare(
    TinyAD
    GIT_REPOSITORY https://github.com/patr-schm/TinyAD.git
    GIT_TAG main
)
FetchContent_MakeAvailable(TinyAD)

# WATT HEADERS
add_library(Watt INTERFACE)
target_include_directories(Watt INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${Eigen3_SOURCE_DIR}
    ${TinyAD_SOURCE_DIR}/include/
)

######################################
############### TESTS ################
######################################
add_subdirectory(Spring)
add_subdirectory(StableNeoHookean)
add_subdirectory(DeformationGradient)
