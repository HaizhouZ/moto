# Find BLASFEO and provide the stable BLASFEO::BLASFEO target.
#
# Search order:
#   1. BLASFEO's own CMake config package (also shipped by conda-forge)
#   2. blasfeo.h and libblasfeo in BLASFEO_ROOT, BLASFEO, CONDA_PREFIX,
#      CMAKE_PREFIX_PATH, and the platform's standard locations

if(NOT TARGET blasfeo)
    find_package(blasfeo CONFIG QUIET)
endif()

if(TARGET blasfeo)
    set(BLASFEO_FOUND TRUE)
    if(NOT BLASFEO_FIND_QUIETLY)
        message(STATUS "Found BLASFEO via config target: blasfeo")
    endif()
    if(NOT TARGET BLASFEO::BLASFEO)
        add_library(BLASFEO::BLASFEO INTERFACE IMPORTED)
        set_property(TARGET BLASFEO::BLASFEO PROPERTY
            INTERFACE_LINK_LIBRARIES blasfeo)
    endif()
    return()
endif()

set(_BLASFEO_HINTS
    ${BLASFEO_ROOT}
    $ENV{BLASFEO_ROOT}
    $ENV{BLASFEO}
    $ENV{CONDA_PREFIX}
)

find_path(BLASFEO_INCLUDE_DIR
    NAMES blasfeo.h blasfeo_common.h
    HINTS ${_BLASFEO_HINTS}
    PATH_SUFFIXES include include/blasfeo)
find_library(BLASFEO_LIBRARY
    NAMES blasfeo
    HINTS ${_BLASFEO_HINTS}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLASFEO
    REQUIRED_VARS BLASFEO_LIBRARY BLASFEO_INCLUDE_DIR)

if(BLASFEO_FOUND AND NOT TARGET BLASFEO::BLASFEO)
    add_library(BLASFEO::BLASFEO UNKNOWN IMPORTED)
    set_target_properties(BLASFEO::BLASFEO PROPERTIES
        IMPORTED_LOCATION "${BLASFEO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLASFEO_INCLUDE_DIR}")
endif()

set(BLASFEO_LIBRARIES "${BLASFEO_LIBRARY}")
set(BLASFEO_INCLUDE_DIRS "${BLASFEO_INCLUDE_DIR}")
mark_as_advanced(BLASFEO_INCLUDE_DIR BLASFEO_LIBRARY)
unset(_BLASFEO_HINTS)
