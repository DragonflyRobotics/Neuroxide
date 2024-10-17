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
