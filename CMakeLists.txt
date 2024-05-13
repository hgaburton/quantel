cmake_minimum_required(VERSION 3.12)

project(quantel)

# Specify the required C++ standard
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# optionally use OpenMP, if so use OpenMP compatible compiler
OPTION(USE_OpenMP "Use OpenMP to enamble <omp.h>" OFF)
OPTION(CODE_COVERAGE "Enable compilation with code coverage flags" OFF)

# Add profiling flags for clang
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   message("-- Enabling code coverage for Clang")
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
   endif()

   if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
       message("-- Enabling code coverage for GCC")
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -g -O0")   
    endif()  

endif(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

# Set source directory
set(SOURCE_DIR "src/quantel")

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})

# Find third-party packages
find_package(Eigen3)
find_package(Boost)
find_package(Libint2)
include_directories(${EIGEN3_INCLUDE_DIRS} ${LIBINT2_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
message(STATUS "EIGEN3_INCLUDE_DIRS: " ${EIGEN3_INCLUDE_DIRS})

# Generate Python module
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(lib/fmt)
pybind11_add_module(quantel ${SOURCES} "${SOURCE_DIR}/bindings.cc"
    "${SOURCE_DIR}/test.cpp"
    )

target_link_libraries(quantel PRIVATE fmt-header-only)

# specify open MP flags
if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(quantel PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif(USE_OpenMP)


#SET(TEST_DIR "tests")
#SET(TESTS ${SOURCES}
#    "${TEST_DIR}/test_main.cc"
#    "${TEST_DIR}/test_math.cc")

# Generate a test executable
#include_directories(lib/catch2/single_include/catch2)
#include_directories(lib/fmt/include)
#add_executable("${PROJECT_NAME}_test" ${TESTS})

#include_directories(${SOURCE_DIR})
#include_directories(lib/catch2/single_include/catch2)
include_directories(lib/fmt/include)

#add_executable("${PROJECT_NAME}_benchmarks"
#    benchmarks/benchmarks.cc
#    "${SOURCE_DIR}/helpers.cc"
#    "${SOURCE_DIR}/find_irrep.cc"
#    "${SOURCE_DIR}/make_gate.cc"
#    "${SOURCE_DIR}/qubit_basis.cc"
#    "${SOURCE_DIR}/circuit.cc"
#    "${SOURCE_DIR}/computer.cc"
#    "${SOURCE_DIR}/gate.cc"
#    "${SOURCE_DIR}/qubit_operator.cc"
#    "${SOURCE_DIR}/qubit_op_pool.cc"
#    "${SOURCE_DIR}/sparse_tensor.cc"
#    "${SOURCE_DIR}/timer.cc")

#target_link_libraries(qforte_benchmarks PRIVATE fmt-header-only)
