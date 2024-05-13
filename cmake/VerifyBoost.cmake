macro(verify_boost VERIFY)

set(EXTRA_LIBRARIES ${ARGV1})
set(CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${Boost_LIBRARIES} ${EXTRA_LIBRARIES})

check_cxx_source_compiles("
#include <iostream>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
int main(int argc, char* argv[]) {
  file_size(\"/dev/null\");
  return 0;
}
" Boost_COMPILES)

if(Boost_COMPILES)
    set(${VERIFY} 1)
else(Boost_COMPILES)
    set(${VERIFY} 0)
endif(Boost_COMPILES)

endmacro(verify_boost)
