

find_package(pybind11 REQUIRED)

# set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})

# @ref from raisim cmakelists
execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
            import sysconfig
            print(sysconfig.get_paths()[\"purelib\"])"
        OUTPUT_VARIABLE PYTHON_SITE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-- python site is ${PYTHON_SITE}")

add_project_dependency(pybind11 REQUIRED)

set(PYTHON_MODULE_NAME ${PROJECT_NAME})
set(BINDING_FILES "")
set(SUBDIRS)

file(GLOB BINDING_SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUBDIR ${BINDING_SUBDIRS})
  if(IS_DIRECTORY ${SUBDIR})
    message(STATUS "Adding binding subdirectory: ${SUBDIR}")
    add_subdirectory(${SUBDIR})
    list(APPEND SUBDIRS ${SUBDIR})
  endif()
endforeach()


foreach(dir IN LISTS SUBDIRS)
        file(GLOB TOP_PY_FILES "${dir}/*.py")
        install(FILES ${TOP_PY_FILES}
                DESTINATION ${PYTHON_SITE}/${PYTHON_MODULE_NAME})
endforeach()