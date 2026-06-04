if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "
import sysconfig
print(sysconfig.get_paths()[\"purelib\"])"
        OUTPUT_VARIABLE PYTHON_SITE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-- python site is ${PYTHON_SITE}")
if("${PYTHON_SITE}" STREQUAL "")
    message(FATAL_ERROR "Could not determine python site-packages directory")
endif()
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "
import sysconfig
print(sysconfig.get_path('purelib', vars={'base': '', 'platbase': ''}).lstrip('/'))"
        OUTPUT_VARIABLE PYTHON_SITE_INSTALL
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if("${PYTHON_SITE_INSTALL}" STREQUAL "")
    message(FATAL_ERROR "Could not determine python install site-packages directory")
endif()
message("-- python install site is ${PYTHON_SITE_INSTALL}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(nanobind CONFIG REQUIRED HINTS "${nanobind_ROOT}")


set(PYTHON_MODULE_NAME ${PROJECT_NAME}_pywrap)
set(BINDING_FILES
  definition/fields.cpp
  definition/functional.cpp
  definition/node_data.cpp
  definition/ns_sqp.cpp
)
if(BUILD_MULTIBODY)
  list(APPEND BINDING_FILES definition/multibody.cpp)
endif()
set(PYTHON_BINDING_FILES
  definition/public_api.py
  definition/sqp.py
  definition/var.py
)

nanobind_add_module(${PYTHON_MODULE_NAME} NB_STATIC ${BINDING_FILES} setup_bindings.cpp)
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE ${PROJECT_NAME} ${PLUGIN_LIBS})
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(BUILD_MULTIBODY)
  target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE MOTO_BUILD_MULTIBODY)
endif()

install(TARGETS ${PYTHON_MODULE_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL}/${PROJECT_NAME})

install(FILES ${PYTHON_BINDING_FILES}
        DESTINATION ${PYTHON_SITE_INSTALL}/${PROJECT_NAME}/definition)

install(FILES package_init.py
        DESTINATION ${PYTHON_SITE_INSTALL}/${PROJECT_NAME}
        RENAME __init__.py)

set(PYTHON_STUB_FILE ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_MODULE_NAME}.pyi)
add_custom_command(
  OUTPUT ${PYTHON_STUB_FILE}
  COMMAND ${CMAKE_COMMAND} -E env
          "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}"
          "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}"
          ${Python_EXECUTABLE} -m nanobind.stubgen
          -q
          -i "${CMAKE_CURRENT_BINARY_DIR}"
          -m ${PYTHON_MODULE_NAME}
          -o ${PYTHON_STUB_FILE}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${PYTHON_MODULE_NAME}
  VERBATIM
)
add_custom_target(${PYTHON_MODULE_NAME}_stub ALL DEPENDS ${PYTHON_STUB_FILE})
install(FILES ${PYTHON_STUB_FILE}
        DESTINATION ${PYTHON_SITE_INSTALL}/${PROJECT_NAME})
