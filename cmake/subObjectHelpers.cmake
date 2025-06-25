set(ALL_SUB_OBJECTS "" )
macro(add_project_sub_object MODULE_NAME)
    # This macro is used to add a sub-object library to the main project.
    # It appends the target objects of the specified module to ALL_SUB_OBJECTS.
    # The MODULE_NAME should be the name of the target object library.
    
    if(NOT TARGET ${MODULE_NAME})
        message(FATAL_ERROR "Target ${MODULE_NAME} does not exist.")
    endif()
    
    message(STATUS "Adding sub-object library: ${MODULE_NAME}")
    # set_property(TARGET ${MODULE_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
    set(ALL_SUB_OBJECTS ${ALL_SUB_OBJECTS};$<TARGET_OBJECTS:${MODULE_NAME}> PARENT_SCOPE)
    target_add_deps(${MODULE_NAME})
endmacro()

macro(target_add_deps TARGET_NAME)
    target_include_directories(
        ${TARGET_NAME} PUBLIC
        ${INCLUDE_DIRS}
    )
    target_compile_options(${TARGET_NAME} PUBLIC ${MOTO_COMPILE_OPTS})
    target_compile_definitions(${TARGET_NAME} PUBLIC ${MOTO_COMPILE_DEFS})
    target_compile_features(${TARGET_NAME} PUBLIC ${MOTO_COMPILE_FEATURES})
    target_link_libraries(${TARGET_NAME} PUBLIC ${MOTO_DEP_LIBS})
endmacro()