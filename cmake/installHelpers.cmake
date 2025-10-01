install(TARGETS ${PROJECT_NAME} ${PLUGIN_LIBS}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
set(ConfigPackageLocation lib/cmake/${PROJECT_NAME})
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/_Config.cmake.in" # <input> template
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"         # <output> generated file
    INSTALL_DESTINATION "${ConfigPackageLocation}"
)
install(EXPORT
    ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${ConfigPackageLocation}
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    # "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${ConfigPackageLocation}
)

# make uninstall
add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)