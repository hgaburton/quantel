cmake_minimum_required(VERSION 3.14)

project(_quantel)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Installation destination for the _quantel extension module.
# Default: install into the source tree so 'import quantel' works from the repo root.
# Override with -DQUANTEL_INSTALL_DIR=<path> for conda/pip packaging
# (e.g. -DQUANTEL_INSTALL_DIR=$SP_DIR/quantel/lib in a conda build script).
set(QUANTEL_INSTALL_DIR "${ROOT_DIR}/quantel/lib" CACHE PATH
    "Directory where the _quantel extension module is installed")
message(STATUS "ROOT_DIR: ${ROOT_DIR}")
message(STATUS "QUANTEL_INSTALL_DIR: ${QUANTEL_INSTALL_DIR}")

# Options
option(CODE_COVERAGE "Enable compilation with code coverage flags" OFF)

# Code coverage flags
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "Enabling code coverage for Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(STATUS "Enabling code coverage for GCC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -g -O0")
    endif()
endif()

# Set source directory
set(SOURCE_DIR "src/")
message(STATUS "SOURCE DIRECTORY: ${SOURCE_DIR}")

# BLAS/LAPACK — on macOS prefer Apple's Accelerate framework
if(APPLE)
    set(BLA_VENDOR Apple)
endif()
find_package(BLAS)
find_package(LAPACK)

# fmt (header-only interface)
find_package(fmt CONFIG REQUIRED)

# libint2
find_package(Libint2 CONFIG REQUIRED)

# Armadillo (header-only, links against the BLAS/LAPACK found above)
# SetupArmadillo.cmake creates an INTERFACE target 'armadillo' from the
# system headers; ARMADILLO_PATH is unset so it searches standard locations.
include(cmake/SetupArmadillo.cmake)

# Third-party packages
find_package(Eigen3 REQUIRED)
find_package(Boost)

# Generate Python module
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_quantel
    "${SOURCE_DIR}/bindings.cpp"
    "${SOURCE_DIR}/shell_pair.cpp"
    "${SOURCE_DIR}/molecule.cpp"
    "${SOURCE_DIR}/libint_interface.cpp"
    "${SOURCE_DIR}/fcidump_interface.cpp"
    "${SOURCE_DIR}/linalg.cpp"
    "${SOURCE_DIR}/determinant.cpp"
    "${SOURCE_DIR}/ci_space.cpp"
    "${SOURCE_DIR}/ci_space2.cpp"
)

target_include_directories(_quantel PRIVATE
    ${SOURCE_DIR}
    ${Boost_INCLUDE_DIRS})

target_link_libraries(_quantel PRIVATE
    fmt::fmt-header-only
    armadillo
    Libint2::int2
    Eigen3::Eigen
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${Boost_LIBRARIES})

# On macOS, link Accelerate explicitly when BLAS/LAPACK come from it
if(APPLE AND BLA_VENDOR STREQUAL "Apple")
    target_link_libraries(_quantel PRIVATE "-framework Accelerate")
endif()

# OpenMP — required.
# Try to find it outright first (works in conda environments with llvm-openmp).
# Fall back to the Homebrew libomp workaround for native macOS Apple Clang builds.
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(LIBOMP_PREFIX)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "Using Homebrew libomp: ${LIBOMP_PREFIX}")
        find_package(OpenMP REQUIRED)
    else()
        message(FATAL_ERROR "OpenMP is required but was not found. "
                            "In a conda environment: conda install llvm-openmp. "
                            "Otherwise: brew install libomp")
    endif()
endif()
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP is required but was not found.")
endif()
target_link_libraries(_quantel PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS _quantel DESTINATION "${QUANTEL_INSTALL_DIR}")
