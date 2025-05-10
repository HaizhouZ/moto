cmake_minimum_required(VERSION 3.5.0)
project(atri VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(magic_enum REQUIRED)
find_package(fmt REQUIRED)
find_package(Doxygen)

if(DOXYGEN_FOUND)
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  add_custom_target(doc_doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    include
)
add_compile_options(-march=native)

file(GLOB SRC_FILES src/*.cpp src/core/*.cpp src/solver/*.cpp src/utils/*.cpp)
add_library(atri SHARED ${SRC_FILES})

add_executable(atri_test main.cpp)
# add_compile_definitions(EIGEN_USE_BLAS)
target_link_libraries(atri PUBLIC 
    # OpenBLAS::OpenBLAS 
    # "/usr/local/blasfeo/lib/libblasfeo.a"
    magic_enum::magic_enum
    fmt::fmt)
target_link_libraries(atri_test PUBLIC
    atri
)

add_subdirectory(example/double_integrator)