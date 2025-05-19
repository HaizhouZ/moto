add_executable(arm_test src/main.cpp)


target_include_directories(arm_test PUBLIC include)
target_link_libraries(arm_test PUBLIC atri)