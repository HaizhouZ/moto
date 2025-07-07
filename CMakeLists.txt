cmake_minimum_required(VERSION 3.10.0)
project(moto)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

include(cmake/depHelpers.cmake)

add_project_dependency(Eigen3 REQUIRED)
add_project_dependency(magic_enum REQUIRED)
add_project_dependency(fmt REQUIRED)
add_project_dependency(OpenMP REQUIRED)
add_project_dependency(CASADI REQUIRED)
add_project_dependency(pybind11 REQUIRED)

option(USE_MIMALLOC "Use mimalloc" OFF)

if(USE_MIMALLOC MATCHES ON)
  message(STATUS "Using mimalloc")
  add_project_dependency(mimalloc REQUIRED)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DEIGEN_MALLOC_ALREADY_ALIGNED=0)
  list(APPEND MOTO_DEP_LIBS mimalloc)
else()
  message(STATUS "Using system malloc")
endif()

include(cmake/docHelpers.cmake)

# set(CMAKE_CXX_STANDARD 20)

set(INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set(MOTO_COMPILE_OPTS -march=native -fPIC)
set(MOTO_COMPILE_DEFS MOTO_USE_OMP)
set(MOTO_COMPILE_FEATURES cxx_std_20)
list(APPEND MOTO_DEP_LIBS
    # OpenBLAS::OpenBLAS 
    # "/usr/local/blasfeo/lib/libblasfeo.a"
    casadi::casadi
    Eigen3::Eigen
    magic_enum::magic_enum
    fmt::fmt
    OpenMP::OpenMP_CXX
    pybind11::embed
)

include(cmake/subObjectHelpers.cmake)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_add_deps(${PROJECT_NAME})

include(cmake/installHelpers.cmake)

add_subdirectory(example)
add_subdirectory(bindings)
add_subdirectory(unittests)
