set(GMP_ROOT "" CACHE PATH "path ")

find_path(GMP_INCLUDE_DIR gmp.h gmpxx.h 
    PATHS $ENV{GMP_ROOT}/include /usr/include /usr/local/include)

find_library(GMP_LIBRARIES NAMES gmp libgmp gmpxx libgmpxx 
    PATHS $ENV{GMP_ROOT}/lib /usr/lib /usr/local/lib)

if(GMP_INCLUDE_DIR AND GMP_LIBRARIES)
    get_filename_component(GMP_LIBRARY_DIR ${GMP_LIBRARIES} PATH)
    set(GMP_FOUND TRUE)
endif()

if(GMP_FOUND)
   if(NOT GMP_FIND_QUIETLY)
       MESSAGE(STATUS "Found GMP: ${GMP_LIBRARIES}")
    endif()
elseif(GMP_FOUND)
    if(GMP_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find GMP")
    endif()
endif()
