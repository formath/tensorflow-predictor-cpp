cmake_minimum_required(VERSION 2.8)

project(deep-ctr)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -msse2 -fPIC -O3 -ggdb -Wall -finline-functions")

message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# Add Subdirectory
add_subdirectory(src)
