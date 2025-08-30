cmake_minimum_required(VERSION 3.10.0)
project(moto)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

include(cmake/depHelpers.cmake)
include(cmake/Findccache.cmake)

add_project_dependency(Eigen3 REQUIRED)
add_project_dependency(magic_enum REQUIRED)
add_project_dependency(fmt REQUIRED)
add_project_dependency(OpenMP REQUIRED)
add_project_dependency(CASADI REQUIRED)
add_project_dependency(pybind11 REQUIRED)
add_project_dependency(re2 REQUIRED)
add_project_dependency(nlohmann_json REQUIRED)
add_project_dependency(OpenSSL REQUIRED)
add_project_dependency(blasfeo REQUIRED)

set(MOTO_COMPILE_OPTS -march=native -fPIC -Wno-interference-size)
set(MOTO_COMPILE_DEFS MOTO_USE_OMP)
set(MOTO_COMPILE_FEATURES cxx_std_20)

option(BUILD_MULTIBODY "Build multibody module" ON)

option(USE_MIMALLOC "Use mimalloc" OFF)

if(USE_MIMALLOC MATCHES ON)
  message(STATUS "Using mimalloc")
  add_project_dependency(mimalloc REQUIRED)
  list(APPEND MOTO_COMPILE_DEFS -DEIGEN_MALLOC_ALREADY_ALIGNED=0)
  list(APPEND MOTO_DEP_LIBS mimalloc-static)
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

list(APPEND MOTO_DEP_LIBS
    # OpenBLAS::OpenBLAS 
    "/usr/local/blasfeo/lib/libblasfeo.a"
    casadi::casadi
    Eigen3::Eigen
    magic_enum::magic_enum
    fmt::fmt
    OpenMP::OpenMP_CXX
    re2::re2
    nlohmann_json::nlohmann_json
    OpenSSL::Crypto
)

include(cmake/subObjectHelpers.cmake)

file(GLOB_RECURSE SRC_FILES src/core/*.cpp src/ocp/*.cpp src/utils/*.cpp src/solver/*.cpp src/spmm/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_add_deps(${PROJECT_NAME})
file(GLOB_RECURSE RELEASE_FILE_LIST src/utils/codegen.cpp)# src/spmm/*.cpp)
set_source_files_properties(${RELEASE_FILE_LIST} PROPERTIES
    COMPILE_OPTIONS "$<$<CONFIG:Debug>:-O3>"
    COMPILE_DEFINITIONS "$<$<CONFIG:Debug>:NDEBUG>"
)
# target_link_options(${PROJECT_NAME} PUBLIC -fsanitize=address)

set(PLUGIN_LIBS "")

if (BUILD_MULTIBODY MATCHES ON)
  message(STATUS "Building multibody module")
  file(GLOB_RECURSE MB_SRC_FILES src/multibody/*.cpp)
  add_library(moto_multibody SHARED ${MB_SRC_FILES})
  add_project_dependency(pinocchio REQUIRED)
  target_add_deps(moto_multibody)
  target_compile_definitions(moto_multibody PUBLIC MOTO_WITH_PINOCCHIO COAL_DISABLE_HPP_FCL_WARNINGS)
  target_link_libraries(moto_multibody PUBLIC ${PROJECT_NAME} pinocchio::pinocchio_parsers)
  list(APPEND PLUGIN_LIBS moto_multibody)
endif()

include(cmake/installHelpers.cmake)

add_subdirectory(example)
add_subdirectory(bindings)

include(CTest)
enable_testing()
add_subdirectory(unittests)
