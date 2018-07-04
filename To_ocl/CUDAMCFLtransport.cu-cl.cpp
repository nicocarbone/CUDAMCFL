#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




/*CU2CL Unhandled -- No main() found
CU2CL Boilerplate inserted here:
CU2CL Initialization:
__cu2cl_Init();


CU2CL Cleanup:
__cu2cl_Cleanup();
*/
cl_kernel __cu2cl_Kernel_MCd;
cl_kernel __cu2cl_Kernel_MCd3D;
cl_kernel __cu2cl_Kernel_LaunchPhoton_Global;
cl_program __cu2cl_Program_CUDAMCFLtransport_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_kernel __cu2cl_Kernel___cu2cl_Memset;
extern cl_program __cu2cl_Util_Program;
extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_CUDAMCFLtransport_cu() {
    clReleaseKernel(__cu2cl_Kernel_MCd);
    clReleaseKernel(__cu2cl_Kernel_MCd3D);
    clReleaseKernel(__cu2cl_Kernel_LaunchPhoton_Global);
    clReleaseProgram(__cu2cl_Program_CUDAMCFLtransport_cu);
}
void __cu2cl_Init_CUDAMCFLtransport_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("CUDAMCFLtransport_cu_cl.aocx", &progSrc);
    __cu2cl_Program_CUDAMCFLtransport_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("CUDAMCFLtransport.cu-cl.cl", &progSrc);
    __cu2cl_Program_CUDAMCFLtransport_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_CUDAMCFLtransport_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_MCd = clCreateKernel(__cu2cl_Program_CUDAMCFLtransport_cu, "MCd", NULL);
    __cu2cl_Kernel_MCd3D = clCreateKernel(__cu2cl_Program_CUDAMCFLtransport_cu, "MCd3D", NULL);
    __cu2cl_Kernel_LaunchPhoton_Global = clCreateKernel(__cu2cl_Program_CUDAMCFLtransport_cu, "LaunchPhoton_Global", NULL);
}

/*	This file is part of CUDAMCFL.

    CUDAMCFL is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCFL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCFL.  If not, see <http://www.gnu.org/licenses/>.
*/

// forward declaration of the device code











//end MCd


//end MCd3D









// end Spin









/*
//Device function to add an unsigned integer to an unsigned long long using CUDA. Needed for Compute Capability 1.1
__device__ void AtomicAddULL(unsigned long long* address, unsigned int add)
{
	if(atomicAdd((unsigned int*)address,add)+add<add)
		atomicAdd(((unsigned int*)address)+1,1u);
}
*/
