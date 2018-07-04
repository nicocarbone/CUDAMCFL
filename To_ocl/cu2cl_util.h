/*
(C) 2010-2017 Virginia Polytechnic Institute & State University (also known as Virginia Tech). All Rights Reserved.his software is provided as-is.  Neither the authors, Virginia Tech nor Virginia Tech Intellectual Properties, Inc. assert, warrant, or guarantee that the software is fit for any purpose whatsoever, nor do they collectively or individually accept any responsibility or liability for any action or activity that results from the use of this software.  The entire risk as to the quality and performance of the software rests with the user, and no remedies shall be provided by the authors, Virginia Tech or Virginia Tech Intellectual Properties, Inc.
 
 This library is free software; you can redistribute it and/or modify it under the terms of the attached GNU Lesser General Public License v2.1 as published by the Free Software Foundation.
 
 This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
*/ 

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
void __cu2cl_Init();

void __cu2cl_Cleanup();
size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc);

cl_int __cu2cl_Memset(cl_mem devPtr, int value, size_t count);

void __cu2cl_Init_CUDAMCFLtransport_cu();

void __cu2cl_Cleanup_CUDAMCFLtransport_cu();


#ifdef __cplusplus
}
#endif
