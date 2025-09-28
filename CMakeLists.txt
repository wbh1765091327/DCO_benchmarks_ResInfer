cmake_minimum_required(VERSION 3.22)
project(Learn-to-Prune)

# set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(include/hnswlib)
include_directories(include/ivf)
include_directories(/usr/include/eigen3)


# find_package(OpenMP)

# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# else()
#     message(FATAL_ERROR "no OpenMP supprot")
# endif()


# add_definitions(-std=c++20 -Ofast -march=core-avx2 -mavx512f -fpic -fopenmp -ftree-vectorize -fexceptions)
add_definitions(-std=c++20 -O3)
add_subdirectory(src)