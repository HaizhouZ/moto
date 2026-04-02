find_package(Catch2 3 QUIET)
if(Catch2_FOUND)
    message(STATUS "Packaged version of Catch will be used.")
else()
    message(STATUS "Bundled version of Catch will be downloaded and used.")
    include(FetchContent)
    FetchContent_Declare(
        Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG 53d0d913a422d356b23dd927547febdf69ee9081 # v3.5.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

find_package(Python 3.8 COMPONENTS Interpreter REQUIRED)

function(add_moto_test name)
    set(test_name ${name}_test)
    set(test_file ${name}.cpp)

    add_executable(${test_name} ${test_file})
    target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${test_name} PRIVATE moto)
    target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain)

    # set_source_files_properties(${test_file} PROPERTIES
    #     COMPILE_OPTIONS "$<$<CONFIG:Debug>:-O3>"
    #     COMPILE_DEFINITIONS "$<$<CONFIG:Debug>:NDEBUG>"
    # )
    
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

function(add_moto_python_test name script)
    add_test(
        NAME ${name}
        COMMAND ${Python_EXECUTABLE} ${script}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(
        ${name}
        PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1"
    )
endfunction()


add_moto_test(sym_test)
add_moto_test(graph_model_compose)
add_moto_test(restoration)
# add_moto_test(expr_ownership)
# add_moto_test(constr_cast)
# add_moto_test(dynamics)
# add_moto_test(field_conversion)
# add_moto_test(spmm)
# add_moto_test(dynamics_cen)
# add_moto_test(blasfeo_factor)
# add_moto_test(multibody_euler)
# target_link_libraries(multibody_euler_test PRIVATE )
