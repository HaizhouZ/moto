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

enable_testing()


function(add_atri_test name)
    set(test_name ${name}_test)
    set(test_file ${name}.cpp)

    add_executable(${test_name} ${test_file})
    target_include_directories(${test_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(${test_name} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain)
    
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

# add_atri_test(external_function)
# add_atri_test(symbolic_comp)