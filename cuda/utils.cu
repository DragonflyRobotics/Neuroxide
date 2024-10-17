#include <cstdio>
#include<cuda_runtime.h>

extern "C" 
int getDeviceName_main(char* name, int device)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    sprintf(name, "%s", prop.name);
    return 0;
}

extern "C"
int getTotalMem_main(size_t* mem, int device)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    *mem = prop.totalGlobalMem;
    return 0;
}
