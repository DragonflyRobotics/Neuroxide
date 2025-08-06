#include <cstdio>
#include<cuda_runtime.h>
#include "handles.cuh"

extern "C" 
int getDeviceName_main(char* name, int device)
{
    cudaDeviceProp prop = handles.prop;
    sprintf(name, "%s", prop.name);
    return 0;
}

extern "C"
int getTotalMem_main(size_t* mem, int device)
{
    cudaDeviceProp prop = handles.prop;
    *mem = prop.totalGlobalMem;
    return 0;
}
