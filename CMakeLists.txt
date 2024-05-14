cmake_minimum_required(VERSION 3.12)

project(_quantel)

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
set(SOURCE_DIR "src/")
message(STATUS "SOURCE DIRECTORY=" ${SOURCE_DIR})
message(STATUS "SOURCE DIRECTORY=" ${CMAKE_CURRENT_SOURCE_DIR})

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})

# Find third-party packages
list(APPEND CMAKE_PREFIX_PATH "lib/")
find_package(Eigen3)
find_package(Boost)
find_package(Libint2)
include_directories(${EIGEN3_INCLUDE_DIRS} ${LIBINT2_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
message(STATUS "EIGEN3_INCLUDE_DIRS: " ${EIGEN3_INCLUDE_DIRS})
find_package(ambit 0.6 REQUIRED)
message(STATUS "AMBIT_INCLUDE_DIRS:" ${AMBIT_INCLUDE_DIRS})

# Generate Python module
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_quantel ${SOURCES} "${SOURCE_DIR}/bindings.cpp"
    "${SOURCE_DIR}/molecule.cpp"
    "${SOURCE_DIR}/libint_interface.cpp"
    )


add_subdirectory(lib/fmt)
target_link_libraries(_quantel PRIVATE fmt-header-only)
target_link_libraries(_quantel PRIVATE
    ${LAPACK_LIBRARIES} ${LIBINT2_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})

# specify open MP flags
if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(_quantel PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif(USE_OpenMP)


install(TARGETS _quantel DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/quantel/lib/")
