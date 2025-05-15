# Add the target for the Python module

set(BINDING_FILES ${CMAKE_CURRENT_SOURCE_DIR}/invoke_interface.cpp)

pybind11_add_module(invoke_interface ${BINDING_FILES})

install(TARGETS invoke_interface
        LIBRARY DESTINATION ${PYTHON_SITE}/${PROJECT_NAME})