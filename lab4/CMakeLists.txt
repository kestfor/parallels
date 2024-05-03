cmake_minimum_required(VERSION 3.22)
project(lab4)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
SET(CMAKE_CXX_STANDARD 17)
add_executable(main main.cpp)

target_link_libraries(main ${MPI_C_LIBRARIES})