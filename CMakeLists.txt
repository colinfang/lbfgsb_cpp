cmake_minimum_required(VERSION 3.10)
project(lbfgsb_cpp CXX Fortran)

set(CMAKE_CXX_STANDARD 17)

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

option(BUILD_EXAMPLE "Build example" OFF)


set(
    FORTRAN_SRC
    Lbfgsb.3.0/lbfgsb.f
    Lbfgsb.3.0/linpack.f
    Lbfgsb.3.0/timer.f
)
set(
    HEADERS
    "include/${PROJECT_NAME}/lbfgsb.hpp"
)

if (NOT TARGET blas)
    set(FORTRAN_SRC ${FORTRAN_SRC}
        Lbfgsb.3.0/blas.f # Now uses external blas
    )
    message(WARNING "lbfgsb_cpp will use local implementation of blas functions")
endif()

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

if (TARGET blas)
    target_link_libraries(${PROJECT_NAME} PUBLIC blas)
endif()

if (CMAKE_BUILD_TYPE MATCHES "Release")
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE -fcheck=all)
endif ()


IF (BUILD_EXAMPLE)
    enable_testing()
    add_subdirectory(examples)
ENDIF()

# Create install rule for your library. Use variables CMAKE_INSTALL_*DIR defined in GNUInstallDir
include(GNUInstallDirs)
# https://cmake.org/cmake/help/latest/command/install.html
# https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT devel
        )

install(FILES ${HEADERS} DESTINATION "include/${PROJECT_NAME}")

install(
    EXPORT ${PROJECT_NAME}_targets
    FILE ${PROJECT_NAME}Config.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)



