

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

set(PYTHON_MODULE_NAME py_${PROJECT_NAME})
set(BINDING_FILES "")

message("=========== PYTHON_MODULE_NAME: ${PYTHON_MODULE_NAME} ===========")

set(SUBDIRS)

file(GLOB BINDING_SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUBDIR ${BINDING_SUBDIRS})
  if(IS_DIRECTORY ${SUBDIR})
    message(STATUS "Adding binding subdirectory: ${SUBDIR}")
    add_subdirectory(${SUBDIR})
    list(APPEND SUBDIRS ${SUBDIR})
  endif()
endforeach()

pybind11_add_module(${PYTHON_MODULE_NAME} ${BINDING_FILES})


install(TARGETS ${PYTHON_MODULE_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE}/${PROJECT_NAME})

install(DIRECTORY ${SUBDIRS}
        DESTINATION ${PYTHON_SITE}/${PYTHON_MODULE_NAME}
        FILES_MATCHING PATTERN "*.py")