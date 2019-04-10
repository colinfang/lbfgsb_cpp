cmake_minimum_required(VERSION 3.10)
project(lbfgsb_cpp CXX Fortran)

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

option(BUILD_EXAMPLE "Build example" OFF)


set(
    FORTRAN_SRC
    "Lbfgsb.3.0/blas.f"
    "Lbfgsb.3.0/lbfgsb.f"
    "Lbfgsb.3.0/linpack.f"
    "Lbfgsb.3.0/timer.f"
)
set(
    HEADERS
    "include/${PROJECT_NAME}/lbfgsb.hpp"
)

add_library(${PROJECT_NAME} ${FORTRAN_SRC} ${HEADERS})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(
    ${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)

target_compile_options(${PROJECT_NAME} PRIVATE -frecursive)

if (CMAKE_BUILD_TYPE MATCHES "Release")
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE -fcheck=all)
endif ()


IF (BUILD_EXAMPLE)
    add_subdirectory(examples)
ENDIF()


install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_targets
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
)

install(FILES ${HEADERS} DESTINATION "include/${PROJECT_NAME}")

install(
    EXPORT ${PROJECT_NAME}_targets
    FILE ${PROJECT_NAME}Config.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)


