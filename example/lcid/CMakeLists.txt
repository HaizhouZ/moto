add_executable(lcid_test src/main.cpp)


target_include_directories(lcid_test PUBLIC include)
target_link_libraries(lcid_test PUBLIC moto moto_multibody)