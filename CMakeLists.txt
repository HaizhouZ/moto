cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME atri)
project(${PROJECT_NAME})

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(magic_enum REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CASADI REQUIRED)
find_package(Boost REQUIRED COMPONENTS regex)

option(USE_MIMALLOC "Use mimalloc" OFF)
option(BUILD_DOCUMENTATION "Build documentation with doxygen" OFF)

set(LIBS)

if(USE_MIMALLOC MATCHES ON)
  message(STATUS "Using mimalloc")
  find_package(mimalloc REQUIRED)
  add_compile_options(-DEIGEN_MALLOC_ALREADY_ALIGNED=0)
  list(APPEND LIBS mimalloc)
else()
  message(STATUS "Using system malloc")
endif()

# if(BUILD_DOCUMENTATION MATCHES ON)
message(STATUS "Building documentation with doxygen")
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

set(CMAKE_CXX_STANDARD 20)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    include
)
add_compile_options(-march=native)
add_compile_definitions(ATRI_USE_OMP)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# add_compile_definitions(EIGEN_USE_BLAS)
message("Linked libraries: ${LIBS}")
target_link_libraries(${PROJECT_NAME} PUBLIC 
    # OpenBLAS::OpenBLAS 
    # "/usr/local/blasfeo/lib/libblasfeo.so"
    ${LIBS}
    casadi
    OpenMP::OpenMP_CXX
    magic_enum::magic_enum
    fmt::fmt)

add_subdirectory(example)
add_subdirectory(bindings)

# install(TARGETS ${PROJECT_NAME}
#   EXPORT atriTargets
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib
#   RUNTIME DESTINATION bin
#   INCLUDES DESTINATION include
# )

# install(DIRECTORY include/ DESTINATION include)

# make uninstall
add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

add_subdirectory(unittests)