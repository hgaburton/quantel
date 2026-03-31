cmake_minimum_required(VERSION 3.12)

project(_quantel)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_PREFIX "${ROOT_DIR}/quantel/lib")
message(STATUS "ROOT_DIR: ${ROOT_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# Options
option(USE_OpenMP "Use OpenMP to enable <omp.h>" OFF)
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

# Armadillo (header-only, links against the BLAS/LAPACK found above)
set(ARMADILLO_PATH "${ROOT_DIR}/external/armadillo")
include(cmake/SetupArmadillo.cmake)

# Third-party packages
find_package(Eigen3 REQUIRED)
find_package(Boost)

# External subprojects
add_subdirectory(external/fmt)
add_subdirectory(external/libint2)

# Generate Python module
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_quantel
    "${SOURCE_DIR}/bindings.cpp"
    "${SOURCE_DIR}/shell_pair.cpp"
    "${SOURCE_DIR}/molecule.cpp"
    "${SOURCE_DIR}/libint_interface.cpp"
    "${SOURCE_DIR}/linalg.cpp"
    "${SOURCE_DIR}/determinant.cpp"
    "${SOURCE_DIR}/ci_space.cpp"
    "${SOURCE_DIR}/ci_space2.cpp"
)

target_include_directories(_quantel PRIVATE
    ${SOURCE_DIR}
    ${Boost_INCLUDE_DIRS})

target_link_libraries(_quantel PRIVATE
    fmt-header-only
    armadillo
    libint2
    Eigen3::Eigen
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${Boost_LIBRARIES})

# On macOS, link Accelerate explicitly when BLAS/LAPACK come from it
if(APPLE AND BLA_VENDOR STREQUAL "Apple")
    target_link_libraries(_quantel PRIVATE "-framework Accelerate")
endif()

# OpenMP — optional, requires special handling on macOS with Apple Clang
if(USE_OpenMP)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Apple Clang doesn't bundle libomp; find it via Homebrew
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
        else()
            message(WARNING "USE_OpenMP=ON but Homebrew libomp not found. "
                            "Install with: brew install libomp")
        endif()
    endif()
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(_quantel PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

install(TARGETS _quantel DESTINATION ${ROOT_DIR}/quantel/lib)
