cmake_minimum_required(VERSION 3.12)

project(_quantel)

include(ExternalProject)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_PREFIX "${ROOT_DIR}/quantel/lib") 
message(STATUS "ROOT_DIR: ${ROOT_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX:" ${CMAKE_INSTALL_PREFIX})

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

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})

# Armadillo
find_package(BLAS)
find_package(LAPACK)
set(ARMADILLO_PATH "${ROOT_DIR}/external/armadillo")
include(cmake/SetupArmadillo.cmake)
include_directories($<TARGET_PROPERTY:armadillo,INTERFACE_INCLUDE_DIRECTORIES>)
set_property(DIRECTORY . APPEND PROPERTY COMPILE_DEFINITIONS
    $<TARGET_PROPERTY:armadillo,INTERFACE_COMPILE_DEFINITIONS>)


# Find third-party packages
#list(APPEND CMAKE_PREFIX_PATH "lib/")
find_package(Eigen3)
find_package(Boost)
include_directories(${EIGEN3_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Generate Python module
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_quantel ${SOURCES} "${SOURCE_DIR}/bindings.cpp"
    "${SOURCE_DIR}/molecule.cpp"
    "${SOURCE_DIR}/libint_interface.cpp"
    "${SOURCE_DIR}/linalg.cpp"
    )

add_subdirectory(external/fmt)
add_subdirectory(external/libint2)


target_link_libraries(_quantel PRIVATE 
            fmt-header-only 
            armadillo 
            libint2 
            ${LAPACK_LIBRARIES}
            ${BLAS_LIBRARIES}
            ${Boost_LIBRARIES} 
            ${EIGEN3_LIBRARIES})

# specify open MP flags
if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(_quantel libint2 PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif(USE_OpenMP)

install(TARGETS _quantel DESTINATION ${ROOT_DIR}/quantel/lib)

#ExternalProject_Add(libint2 
#    PREFIX ${ROOT_DIR}/external/
#    URL https://github.com/evaleev/libint/releases/download/v2.9.0/libint-2.9.0.tgz
#    DOWNLOAD_DIR ${ROOT_DIR}/external
#    DOWNLOAD_NAME libint2
#    STAMP_DIR ${ROOT_DIR}/external/libint-2.9.0/stamp
#    BINARY_DIR ${ROOT_DIR}/external/libint-2.9.0/build
#    SOURCE_DIR ${ROOT_DIR}/external/libint-2.9.0/libint-2.9.0
#    INSTALL_DIR ${ROOT_DIR}/quantel/lib
#)
#add_dependencies(_quantel libint2)
