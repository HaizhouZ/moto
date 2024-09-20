cmake_minimum_required(VERSION 3.5.0)
project(manbo VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(magic_enum REQUIRED)
find_package(fmt REQUIRED)

set(CMAKE_CXX_STANDARD 20)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    include
)
add_executable(manbo main.cpp test.cpp)
add_compile_definitions(EIGEN_USE_BLAS)
add_compile_options(-march=native )
target_link_libraries(manbo PUBLIC 
    OpenBLAS::OpenBLAS 
    magic_enum::magic_enum
    fmt::fmt)