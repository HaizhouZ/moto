cmake_minimum_required(VERSION 3.10.0)

set(PROJECT_NAMESPACE machines-in-motion)
set(PROJECT_NAME atri)
set(PROJECT_DESCRIPTION "Trajectory Optimization Framework.")
set(PROJECT_URL "https://github.com/HaizhouZ/${PROJECT_NAME}")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT TRUE)

# check jrl modules
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  else()
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n")
  endif()
endif()

include("${JRL_CMAKE_MODULES}/base.cmake")
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE OFF)
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

add_project_dependency(Eigen3 REQUIRED)
add_project_dependency(OpenBLAS REQUIRED)
add_project_dependency(magic_enum REQUIRED)
add_project_dependency(fmt REQUIRED)
add_project_dependency(OpenMP REQUIRED)

option(USE_MIMALLOC "Use mimalloc" OFF)

set(USE_MIMALLOC OFF)

set(LIBS)

if(USE_MIMALLOC MATCHES ON)
  message(STATUS "Using mimalloc")
  find_package(mimalloc REQUIRED)
  add_compile_options(-DEIGEN_MALLOC_ALREADY_ALIGNED=0)
  list(APPEND LIBS mimalloc)
else()
  message(STATUS "using system malloc")
endif()

set(CMAKE_CXX_STANDARD 20)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    include
)
add_compile_options(-march=native)
add_compile_definitions(ATRI_USE_OMP)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

add_executable(atri_test main.cpp)
# add_compile_definitions(EIGEN_USE_BLAS)
message("Linked libraries: ${LIBS}")
target_link_libraries(${PROJECT_NAME} PUBLIC 
    # OpenBLAS::OpenBLAS 
    # "/usr/local/blasfeo/lib/libblasfeo.so"
    ${LIBS}
    OpenMP::OpenMP_CXX
    magic_enum::magic_enum
    fmt::fmt)
target_link_libraries(atri_test PUBLIC
    ${PROJECT_NAME}
)

add_subdirectory(example/double_integrator)
add_subdirectory(bindings)

install(TARGETS ${PROJECT_NAME}
  EXPORT atriTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)