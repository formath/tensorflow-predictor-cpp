cmake_minimum_required(VERSION 2.8)

project(tensorflow-predictor-cpp)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()
if (LINUX)
	SET(LDFLAGS "-std=c++11 -msse4.1 -fPIC -O3 -ggdb -Wall -finline-functions -Wl,--allow-multiple-definition -Wl,--whole-archive")
else()
	SET(LDFLAGS "-std=c++11 -msse4.1 -fPIC -O3 -ggdb -Wall -finline-functions -undefined dynamic_lookup -all_load")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LDFLAGS}")

message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# Add Subdirectory
add_subdirectory(src)
