cmake_minimum_required(VERSION 3.12)
project(code_opt_dev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/kernel/*.cpp"
)

add_library(kernels STATIC ${KERNEL_SOURCES})

target_include_directories(kernels
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)

target_compile_options(kernels
    PRIVATE
        -O2 -ffast-math -Wall -g -fno-omit-frame-pointer
)

# ---- Extra compiler flags for Bonus Tasks ----

# for bitwise.cpp, we use AVX2 instructions
set_source_files_properties(
    "${CMAKE_SOURCE_DIR}/src/kernel/bitwise.cpp"
    PROPERTIES
    COMPILE_OPTIONS "-O3;-mavx2;-march=native;"
)

set_source_files_properties(
    "${CMAKE_SOURCE_DIR}/src/kernel/relu.cpp"
    PROPERTIES
    COMPILE_OPTIONS "-O3;-mavx2;-march=native;"
)

set_source_files_properties(
    "${CMAKE_SOURCE_DIR}/src/kernel/matmul.cpp"
    PROPERTIES
    COMPILE_OPTIONS "-O3;-mavx2;-march=native;"
)

# blackscholes.cpp: per-option loop is parallelised with OpenMP and the LUT/Taylor
# helpers benefit from FMA + AVX2 codegen.
set_source_files_properties(
    "${CMAKE_SOURCE_DIR}/src/kernel/blackscholes.cpp"
    PROPERTIES
    COMPILE_OPTIONS "-O3;-mavx2;-march=native;"
)

target_link_libraries(kernels
    PUBLIC
        OpenMP::OpenMP_CXX
)


target_include_directories(kernels PUBLIC include)

# ---- Executables ----

add_executable(single_bench src/main/single_bench.cpp)
target_link_libraries(single_bench PRIVATE kernels)

add_executable(run_all src/main/run_all.cpp)
target_link_libraries(run_all PRIVATE kernels)
