

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

find_package(pybind11 REQUIRED)

set(PYTHON_MODULE_NAME ${PROJECT_NAME}_pywrap)
set(BINDING_FILES "")
set(SUBDIRS)

file(GLOB BINDING_SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUBDIR ${BINDING_SUBDIRS})
  if(IS_DIRECTORY ${SUBDIR})
    message(STATUS "Adding binding subdirectory: ${SUBDIR}")
    file(GLOB BINDING_FILES_LOCAL ${SUBDIR}/*.cpp)
    list(APPEND BINDING_FILES ${BINDING_FILES_LOCAL})
  endif()
endforeach()

pybind11_add_module(${PYTHON_MODULE_NAME} ${BINDING_FILES} setup_bindings.cpp)

install(TARGETS ${PYTHON_MODULE_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE}/${PROJECT_NAME})

foreach(dir IN LISTS BINDING_SUBDIRS)
        file(GLOB TOP_PY_FILES "${dir}/*.py")
        get_filename_component(DIR_NAME ${dir} NAME)
        install(FILES ${TOP_PY_FILES}
                DESTINATION ${PYTHON_SITE}/${PROJECT_NAME}/${DIR_NAME})
endforeach()

install(FILES __init__.py common.py
        DESTINATION ${PYTHON_SITE}/${PROJECT_NAME})