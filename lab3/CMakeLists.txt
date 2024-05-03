cmake_minimum_required(VERSION 3.22)
project(lab3)

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
SET(CMAKE_C_STANDARD 17)
add_executable(main main.c)

target_link_libraries(main ${MPI_C_LIBRARIES})