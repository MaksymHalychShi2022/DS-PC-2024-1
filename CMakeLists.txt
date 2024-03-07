cmake_minimum_required(VERSION 3.22)
project(DS-PC-2024-1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

add_executable(lab_1 lab_1/main.cpp)
add_executable(lab_2 lab_2/main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab_1 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(lab_2 PUBLIC OpenMP::OpenMP_CXX)
endif()