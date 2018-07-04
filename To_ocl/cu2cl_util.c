 (C) 2010-2017 Virginia Polytechnic Institute & State University (also known as Virginia Tech). All Rights Reserved.
 This software is provided as-is.  Neither the authors, Virginia Tech nor Virginia Tech Intellectual Properties, Inc. assert, warrant, or guarantee that the software is fit for any purpose whatsoever, nor do they collectively or individually accept any responsibility or liability for any action or activity that results from the use of this software.  The entire risk as to the quality and performance of the software rests with the user, and no remedies shall be provided by the authors, Virginia Tech or Virginia Tech Intellectual Properties, Inc.
 
 This library is free software; you can redistribute it and/or modify it under the terms of the attached GNU Lesser General Public License v2.1 as published by the Free Software Foundation.
 
 This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
 
#include "cu2cl_util.h"
extern cl_kernel __cu2cl_Kernel_MCd;
extern cl_kernel __cu2cl_Kernel_MCd3D;
extern cl_kernel __cu2cl_Kernel_LaunchPhoton_Global;
extern cl_program __cu2cl_Program_CUDAMCFLtransport_cu;
const char *progSrc;
size_t progLen;

cl_kernel __cu2cl_Kernel___cu2cl_Memset;
cl_program __cu2cl_Util_Program;
cl_platform_id __cu2cl_Platform;
cl_device_id __cu2cl_Device;
cl_context __cu2cl_Context;
cl_command_queue __cu2cl_CommandQueue;

size_t globalWorkSize[3];
size_t localWorkSize[3];
size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc) {
    FILE *f = fopen(filename, "r");
    fseek(f, 0, SEEK_END);
    size_t len = (size_t) ftell(f);
    *progSrc = (const char *) malloc(sizeof(char)*len);
    rewind(f);
    fread((void *) *progSrc, len, 1, f);
    fclose(f);
    return len;
}


cl_int __cu2cl_Memset(cl_mem devPtr, int value, size_t count) {
    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 0, sizeof(cl_mem), &devPtr);
    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 1, sizeof(cl_uchar), &value);
    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 2, sizeof(cl_uint), &count);
    globalWorkSize[0] = count;
    return clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel___cu2cl_Memset, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
}


void __cu2cl_Init() {
    clGetPlatformIDs(1, &__cu2cl_Platform, NULL);
    clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_ALL, 1, &__cu2cl_Device, NULL);
    __cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);
    __cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);
    __cu2cl_Init_CUDAMCFLtransport_cu();
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("cu2cl_util.aocx", &progSrc);
    __cu2cl_Util_Program = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("cu2cl_util.cl", &progSrc);
    __cu2cl_Util_Program = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Util_Program, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel___cu2cl_Memset = clCreateKernel(__cu2cl_Util_Program, "__cu2cl_Memset", NULL);
}

void __cu2cl_Cleanup() {
    clReleaseKernel(__cu2cl_Kernel___cu2cl_Memset);
    clReleaseProgram(__cu2cl_Util_Program);
    __cu2cl_Cleanup_CUDAMCFLtransport_cu();
    clReleaseCommandQueue(__cu2cl_CommandQueue);
    clReleaseContext(__cu2cl_Context);
}
