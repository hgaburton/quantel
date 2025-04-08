#include <omp.h>

struct omp_device
{
    /// This structure holds information about the OpenMP device
    /// It is used to store the number of threads and the thread ID
    /// for each thread in the OpenMP parallel region
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    bool omp = true;
#else
    int nthreads = 1;
    bool omp = false;
#endif

    /// This function returns the thread id
    int thread_id() 
    {
        #ifdef _OPENMP
            return omp_get_thread_num();
        #else
            return 0;
        #endif
    }
};