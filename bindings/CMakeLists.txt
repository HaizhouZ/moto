if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
        COMMAND "python" -c "
import sysconfig
print(sysconfig.get_paths()[\"purelib\"])"
        OUTPUT_VARIABLE PYTHON_SITE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-- python site is ${PYTHON_SITE}")
if("${PYTHON_SITE}" STREQUAL "")
    message(FATAL_ERROR "Could not determine python site-packages directory")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 OLD)
endif()

find_package(nanobind CONFIG REQUIRED HINTS "${nanobind_ROOT}")


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

list(APPEND BINDING_FILES ${CMAKE_CURRENT_SOURCE_DIR}/definition/model.cpp)

nanobind_add_module(${PYTHON_MODULE_NAME} NB_STATIC ${BINDING_FILES} setup_bindings.cpp)
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE ${PROJECT_NAME} ${PLUGIN_LIBS})
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${PYTHON_MODULE_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE}/${PROJECT_NAME})

foreach(dir IN LISTS BINDING_SUBDIRS)
        file(GLOB TOP_PY_FILES "${dir}/*.py" "${dir}/*.pyi")
        get_filename_component(DIR_NAME ${dir} NAME)
        install(FILES ${TOP_PY_FILES}
                DESTINATION ${PYTHON_SITE}/${PROJECT_NAME}/${DIR_NAME})
endforeach()

install(FILES __init__.py
        DESTINATION ${PYTHON_SITE}/${PROJECT_NAME})

nanobind_add_stub(
  ${PYTHON_MODULE_NAME}_stub
  MODULE ${PYTHON_MODULE_NAME}
  OUTPUT ${PYTHON_SITE}/${PROJECT_NAME}/${PYTHON_MODULE_NAME}.pyi
  PYTHON_PATH ${PYTHON_SITE}/${PROJECT_NAME}
  DEPENDS ${PYTHON_MODULE_NAME}
)
