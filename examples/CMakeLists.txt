add_executable(example example.cpp)
target_link_libraries(example PRIVATE ${PROJECT_NAME})
target_compile_options(
    example PRIVATE
    -Wall
    -Wextra
    -pedantic
    -Wshadow
    -Wdouble-promotion
    -Wuseless-cast
    -Wnull-dereference
    -Wduplicated-branches
    -Wduplicated-cond
)
