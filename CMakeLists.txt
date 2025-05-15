cmake_minimum_required(VERSION 3.5.0)
project(atri VERSION 0.1.0 LANGUAGES C CXX)

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
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n")
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare("jrl-cmakemodules"
                         GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(magic_enum REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Doxygen)

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
add_library(atri SHARED ${SRC_FILES})

add_executable(atri_test main.cpp)
# add_compile_definitions(EIGEN_USE_BLAS)
message("Linked libraries: ${LIBS}")
target_link_libraries(atri PUBLIC 
    # OpenBLAS::OpenBLAS 
    # "/usr/local/blasfeo/lib/libblasfeo.so"
    ${LIBS}
    OpenMP::OpenMP_CXX
    magic_enum::magic_enum
    fmt::fmt)
target_link_libraries(atri_test PUBLIC
    atri
)

add_subdirectory(example/double_integrator)
add_subdirectory(binding)

install(TARGETS atri
  EXPORT atriTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT atriTargets
  FILE atriTargets.cmake
  NAMESPACE atri::
  DESTINATION lib/cmake/atri
)

# install(FILES
#   "${CMAKE_CURRENT_BINARY_DIR}/atriConfig.cmake"
#   "${CMAKE_CURRENT_BINARY_DIR}/atriConfigVersion.cmake"
#   DESTINATION lib/cmake/atri
# )