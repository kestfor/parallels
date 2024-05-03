cmake_minimum_required(VERSION 3.26)
project(bpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")
add_executable(main main.cpp
        Vector.h
        Vector.cpp
        Matrix.cpp)