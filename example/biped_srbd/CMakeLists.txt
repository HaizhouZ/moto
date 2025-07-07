add_executable(biped_srbd_test src/main.cpp)


target_include_directories(biped_srbd_test PUBLIC include)
target_link_libraries(biped_srbd_test PUBLIC moto)