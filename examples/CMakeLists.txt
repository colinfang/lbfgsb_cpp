if (INNER_BUILD)
    add_executable(example example.cpp strncmp_debug.cpp)
    target_link_libraries(example PRIVATE ${PROJECT_NAME})
    target_link_libraries(example PRIVATE lbfgsb::lbfgsb)
    #target_compile_options(
    #    example PRIVATE
    #    -Wall
    #    -Wextra
    #    -pedantic
    #    -Wshadow
    #    -Wdouble-promotion
    #    -Wnull-dereference
    #)

    add_test(NAME example COMMAND example)
else ()
    ExternalProject_Get_property(main binary_dir)
    set(main_BINARY_DIR ${binary_dir})
    configure_file("${CMAKE_SOURCE_DIR}/examples/CTestCustom.cmake.in" ${CMAKE_BINARY_DIR}/CTestCustom.cmake)
    add_test(NAME run-subtests-trigger COMMAND true) # requires at least one even, a dummy one
    
    add_custom_target(run-subtests
            COMMAND ${CMAKE_COMMAND} -E echo "To run custom ctest, run it in ${main_BINARY_DIR}"
            # COMMAND ${CMAKE_COMMAND} -E echo CWD=${CMAKE_BINARY_DIR}
            # COMMAND ${CMAKE_COMMAND} -E echo CMD=${CMAKE_CTEST_COMMAND} -C $<CONFIG>
            # COMMAND ${CMAKE_COMMAND} -E echo ----------------------------------
            COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1 ${CMAKE_CTEST_COMMAND} -C $<CONFIG>
            WORKING_DIRECTORY ${main_BINARY_DIR}
            )
endif ()