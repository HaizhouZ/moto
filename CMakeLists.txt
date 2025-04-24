cmake_minimum_required(VERSION 3.5.0)
project(atri VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(magic_enum REQUIRED)
find_package(fmt REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    include
)
add_compile_options(-march=native)
add_executable(atri main.cpp test.cpp 
    src/expression_base.cpp
    src/shooting_node.cpp
    src/approx.cpp)
# add_compile_definitions(EIGEN_USE_BLAS)
target_link_libraries(atri PUBLIC 
    # OpenBLAS::OpenBLAS 
    "/usr/local/blasfeo/lib/libblasfeo.a"
    magic_enum::magic_enum
    fmt::fmt)