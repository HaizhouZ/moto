cmake_minimum_required(VERSION 3.10.0)
project(atri)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

include(cmake/depHelpers.cmake)

add_project_dependency(Eigen3 REQUIRED)
add_project_dependency(magic_enum REQUIRED)
add_project_dependency(fmt REQUIRED)
add_project_dependency(OpenMP REQUIRED)
add_project_dependency(CASADI REQUIRED)
add_project_dependency(Boost REQUIRED COMPONENTS regex)

option(USE_MIMALLOC "Use mimalloc" OFF)

set(ATRI_PUBLIC_DEPENDENCIES)

if(USE_MIMALLOC MATCHES ON)
  message(STATUS "Using mimalloc")
  add_project_dependency(mimalloc REQUIRED)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -DEIGEN_MALLOC_ALREADY_ALIGNED=0)
  list(APPEND LIBS mimalloc)
else()
  message(STATUS "Using system malloc")
endif()

include(cmake/docHelpers.cmake)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(${PROJECT_NAME} PUBLIC -march=native)
target_compile_definitions(${PROJECT_NAME} PUBLIC ATRI_USE_OMP)
# target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_USE_BLAS)

message("Additionally Linked libraries: ${LIBS}")
target_link_libraries(${PROJECT_NAME} PUBLIC 
    # OpenBLAS::OpenBLAS 
    # "/usr/local/blasfeo/lib/libblasfeo.a"
    ${LIBS}
    casadi::casadi
    OpenMP::OpenMP_CXX
    magic_enum::magic_enum
    fmt::fmt)

add_subdirectory(example)
add_subdirectory(bindings)

add_subdirectory(unittests)

include(cmake/installHelpers.cmake)