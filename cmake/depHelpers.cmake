set(GENERATED_DEPENDENCIES "")

# =====================================================================
# Macro: add_project_dependency
# ---------------------------------------------------------------------
# This macro adds a public dependency to the project.
# It handles both the internal `find_package` for building this project
# and stores arguments for the exported `find_dependency` in the config file.
#
# Usage:
#   add_project_dependency(<PackageName> [VERSION <major>[.minor[.patch[.tweak]]]]
#                          [EXACT] [COMPONENTS <comp1>...] [CONFIG|MODULE]
#                          [REQUIRED|QUIET] # <-- These flags are now explicitly handled
#                          [NO_CMAKE_FRAMEWORK_PATH] [NO_CMAKE_ENVIRONMENT_PATH]
#                          [NO_SYSTEM_ENVIRONMENT_PATH] [NO_CMAKE_PATH]
#                          [NO_SYSTEM_PATH] [NO_DEFAULT_PATH]
#                          [PATHS <path1>...] [PATH_SUFFIXES <suffix1>...]
#                          [NO_POLICY_SCOPE])
#
# All arguments provided are passed through. If neither REQUIRED nor QUIET
# is specified for the internal find_package, it defaults to REQUIRED.
# =====================================================================
macro(add_project_dependency)
    # _internal_find_args for find_package in THIS project.
    # _export_find_args for find_dependency in the GENERATED config file.
    set(_internal_find_args "")
    set(_export_find_args "")

    set(_explicit_required_passed FALSE)
    set(_explicit_quiet_passed FALSE)

    # Parse arguments for both internal and export lists
    foreach(_arg ${ARGV})
        if ("${_arg}" STREQUAL "REQUIRED")
            set(_explicit_required_passed TRUE)
            list(APPEND _export_find_args "REQUIRED") # Pass REQUIRED to export
        elseif ("${_arg}" STREQUAL "QUIET")
            set(_explicit_quiet_passed TRUE)
            list(APPEND _export_find_args "QUIET") # Pass QUIET to export
        else()
            # All other arguments (package name, COMPONENTS, VERSION, CONFIG/MODULE, etc.)
            list(APPEND _internal_find_args "${_arg}")
            list(APPEND _export_find_args "${_arg}")
        endif()
    endforeach()

    string(REPLACE ";" " " _export_find_args "${_export_find_args}")

    # For the internal find_package call, if neither REQUIRED nor QUIET was specified,
    # we default to REQUIRED.
    if(NOT _explicit_required_passed AND NOT _explicit_quiet_passed)
        list(APPEND _internal_find_args REQUIRED)
    endif()

    # Call find_package in the current project to make the targets available.
    find_package(${_internal_find_args})

    # Store the arguments for later generation in atriConfig.cmake.in.
    # GENERATED_DEPENDENCIES expects each item to be a single string
    # representing the full argument list for find_dependency.
    list(APPEND GENERATED_DEPENDENCIES "${_export_find_args}")
endmacro()