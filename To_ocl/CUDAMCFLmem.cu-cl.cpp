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
extern cl_kernel __cu2cl_Kernel_MCd;
extern cl_kernel __cu2cl_Kernel_MCd3D;
extern cl_kernel __cu2cl_Kernel_LaunchPhoton_Global;
extern cl_program __cu2cl_Program_CUDAMCFLtransport_cu;
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

int CopyDeviceToHostMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{ // Copy data from Device to Host memory

	const int xy_size = sim->det.nx + sim->det.ny*sim->det.nx;
	const int num_x=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_y=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_z=(int)((sim->esp)*(double)sim->grid_size);
	const int fhd_size = num_x * num_y * num_z;

	const int num_x_tdet = sim->det.x_temp_numdets;
  const int num_y_tdet = sim->det.y_temp_numdets;
  const long num_tbins = sim->det.temp_bins;
  const long timegrid_size = num_x_tdet * num_y_tdet * num_tbins;

	// Copy Rd_xy, Tt_xy and A_xyz
	clEnqueueReadBuffer(__cu2cl_CommandQueue, DeviceMem->Rd_xy, CL_TRUE, 0, xy_size*sizeof(unsigned long long), HostMem->Rd_xy, 0, NULL, NULL);
	clEnqueueReadBuffer(__cu2cl_CommandQueue, DeviceMem->Tt_xy, CL_TRUE, 0, xy_size*sizeof(unsigned long long), HostMem->Tt_xy, 0, NULL, NULL);

	// Copy fhd
	clEnqueueReadBuffer(__cu2cl_CommandQueue, DeviceMem->fhd, CL_TRUE, 0, fhd_size*sizeof(unsigned long long), HostMem->fhd, 0, NULL, NULL);

	// Copy time array
	clEnqueueReadBuffer(__cu2cl_CommandQueue, DeviceMem->time_xyt, CL_TRUE, 0, timegrid_size*sizeof(unsigned long long), HostMem->time_xyt, 0, NULL, NULL);

	// Copy the state of the RNG's
	clEnqueueReadBuffer(__cu2cl_CommandQueue, DeviceMem->x, CL_TRUE, 0, NUM_THREADS*sizeof(unsigned long long), HostMem->x, 0, NULL, NULL);

	return 0;
}


int InitDCMem(SimulationStruct* sim)
{
	const int num_x=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_y=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_z=(int)((sim->esp)*(double)sim->grid_size);
	const int fhd_size = num_x * num_y * num_z;


	// Copy fhd flag
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, fhd_activated_dc, CL_TRUE, 0, sizeof(unsigned int), &(sim->fhd_activated), 0, NULL, NULL);

	// Copy bulk method flag
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, bulk_method_dc, CL_TRUE, 0, sizeof(unsigned int), &(sim->bulk_method), 0, NULL, NULL);

	// Copy time sim flag
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, do_temp_sim_dc, CL_TRUE, 0, sizeof(unsigned int), &(sim->do_temp_sim), 0, NULL, NULL);


	// Copy det-data to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, det_dc, CL_TRUE, 0, sizeof(DetStruct), &(sim->det), 0, NULL, NULL);

	// Copy inclusion data to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, inclusion_dc, CL_TRUE, 0, sizeof(IncStruct), &(sim->inclusion), 0, NULL, NULL);

	// Copy number of layers to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, n_layers_dc, CL_TRUE, 0, sizeof(unsigned int), &(sim->n_layers), 0, NULL, NULL);

	// Copy number of bulk descriptors to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, n_bulks_dc, CL_TRUE, 0, sizeof(unsigned int), &(sim->n_bulks), 0, NULL, NULL);

	// Copy start_weight_dc to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, start_weight_dc, CL_TRUE, 0, sizeof(unsigned int), &(sim->start_weight), 0, NULL, NULL);

	// Copy grid_size_dc to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, grid_size_dc, CL_TRUE, 0, sizeof(unsigned int), &(sim->grid_size), 0, NULL, NULL);

	// Copy layer data to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, layers_dc, CL_TRUE, 0, (sim->n_layers+2)*sizeof(LayerStruct), sim->layers, 0, NULL, NULL);

	// Copy bulk data to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, bulks_dc, CL_TRUE, 0, (sim->n_bulks+2)*sizeof(BulkStruct), sim->bulks, 0, NULL, NULL);

	// Copy num_photons_dc to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, num_photons_dc, CL_TRUE, 0, sizeof(unsigned long long), &(sim->number_of_photons), 0, NULL, NULL);

	// Copy x source position to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, xi_dc, CL_TRUE, 0, sizeof(float), &(sim->xi), 0, NULL, NULL);

	// Copy y source position to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, yi_dc, CL_TRUE, 0, sizeof(float), &(sim->yi), 0, NULL, NULL);

	// Copy z source position to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, zi_dc, CL_TRUE, 0, sizeof(float), &(sim->zi), 0, NULL, NULL);

	// Copy source direction to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, dir_dc, CL_TRUE, 0, sizeof(float), &(sim->dir), 0, NULL, NULL);

	// Copy esp to constant device memory
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, esp_dc, CL_TRUE, 0, sizeof(float), &(sim->esp), 0, NULL, NULL);

	return 0;

}

int InitMemStructs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{
	const int xy_size = sim->det.nx + sim->det.ny*sim->det.nx; //TODO: more efficient space usage

	const int num_x=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_y=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_z=(int)((sim->esp)*(double)sim->grid_size);
	const int fhd_size = num_x * num_y * num_z;

	const int num_x_tdet = sim->det.x_temp_numdets;
  const int num_y_tdet = sim->det.y_temp_numdets;
  const long num_tbins = sim->det.temp_bins;
  const long timegrid_size = num_x_tdet * num_y_tdet * num_tbins;

	// Allocate p on the device
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->p = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, NUM_THREADS*sizeof(PhotonStruct), NULL, NULL);

	// Allocate Rd_xy on CPU and GPU
	HostMem->Rd_xy = (unsigned long long*) malloc(xy_size*sizeof(unsigned long long));
	if(HostMem->Rd_xy==NULL){printf("Error allocating HostMem->Rd_xy"); exit (1);}
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->Rd_xy = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, xy_size*sizeof(unsigned long long), NULL, NULL);
	__cu2cl_Memset(DeviceMem->Rd_xy, 0, xy_size*sizeof(unsigned long long));

	// Allocate Tt_xy on CPU and GPU
	HostMem->Tt_xy = (unsigned long long*) malloc(xy_size*sizeof(unsigned long long));
	if(HostMem->Tt_xy==NULL){printf("Error allocating HostMem->Tt_xy"); exit (1);}
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->Tt_xy = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, xy_size*sizeof(unsigned long long), NULL, NULL);
	__cu2cl_Memset(DeviceMem->Tt_xy, 0, xy_size*sizeof(unsigned long long));

	// Allocate fhd on CPU and GPU
	HostMem->fhd = (unsigned long long*) malloc(fhd_size*sizeof(unsigned long long));
	if(HostMem->fhd==NULL){printf("Error allocating HostMem->fhd"); exit (1);}
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->fhd = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, fhd_size*sizeof(unsigned long long), NULL, NULL);
	__cu2cl_Memset(DeviceMem->fhd, 0, fhd_size*sizeof(unsigned long long));

	// Allocate timegrid on CPU and GPU
	HostMem->time_xyt = (unsigned long long*) malloc(timegrid_size*sizeof(unsigned long long));
	if(HostMem->time_xyt==NULL){printf("Error allocating HostMem->time_xyt"); exit (1);}
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->time_xyt = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, timegrid_size*sizeof(unsigned long long), NULL, NULL);
	__cu2cl_Memset(DeviceMem->time_xyt, 0, timegrid_size*sizeof(unsigned long long));

	// Allocate x time detectors
	HostMem->tdet_pos_x = (float*) malloc(num_x_tdet*sizeof(float));
	if(HostMem->tdet_pos_x==NULL){printf("Error allocating HostMem->tdet_pos_x"); exit (1);}
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->tdet_pos_x = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, num_x_tdet*sizeof(float), NULL, NULL);
	__cu2cl_Memset(DeviceMem->tdet_pos_x, 0, num_x_tdet*sizeof(float));

	// Allocate y time detectors
	HostMem->tdet_pos_y = (float*) malloc(num_y_tdet*sizeof(float));
	if(HostMem->tdet_pos_y==NULL){printf("Error allocating HostMem->tdet_pos_y"); exit (1);}
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->tdet_pos_y = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, num_y_tdet*sizeof(float), NULL, NULL);
	__cu2cl_Memset(DeviceMem->tdet_pos_y, 0, num_y_tdet*sizeof(float));

	// Allocate x and a on the device (For MWC RNG)
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
  *(void**)&DeviceMem->x = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, NUM_THREADS*sizeof(unsigned long long), NULL, NULL);
  clEnqueueWriteBuffer(__cu2cl_CommandQueue, DeviceMem->x, CL_TRUE, 0, NUM_THREADS*sizeof(unsigned long long), HostMem->x, 0, NULL, NULL);

/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
  *(void**)&DeviceMem->a = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, NUM_THREADS*sizeof(unsigned int), NULL, NULL);
  clEnqueueWriteBuffer(__cu2cl_CommandQueue, DeviceMem->a, CL_TRUE, 0, NUM_THREADS*sizeof(unsigned int), HostMem->a, 0, NULL, NULL);

	// Allocate bulk_info 3D matrix and copy to device memory
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->bulk_info = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, fhd_size*sizeof(short), NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, DeviceMem->bulk_info, CL_TRUE, 0, fhd_size*sizeof(short), sim->bulk_info, 0, NULL, NULL);


	// Allocate thread_active on the device and host
	HostMem->thread_active = (unsigned int*) malloc(NUM_THREADS*sizeof(unsigned int));
	if(HostMem->thread_active==NULL){printf("Error allocating HostMem->thread_active"); exit (1);}
	for(int i=0;i<NUM_THREADS;i++)HostMem->thread_active[i]=1u;
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->thread_active = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, NUM_THREADS*sizeof(unsigned int), NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, DeviceMem->thread_active, CL_TRUE, 0, NUM_THREADS*sizeof(unsigned int), HostMem->thread_active, 0, NULL, NULL);

	//Allocate num_launched_photons on the device and host
	HostMem->num_terminated_photons = (unsigned long long*) malloc(sizeof(unsigned long long));
	if(HostMem->num_terminated_photons==NULL){printf("Error allocating HostMem->num_terminated_photons"); exit (1);}
	*HostMem->num_terminated_photons=0;
/*CU2CL Note -- Identified member expression in cudaMalloc device pointer*/
/*CU2CL Note -- Rewriting single decl*/
	*(void**)&DeviceMem->num_terminated_photons = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(unsigned long long), NULL, NULL);
	clEnqueueWriteBuffer(__cu2cl_CommandQueue, DeviceMem->num_terminated_photons, CL_TRUE, 0, sizeof(unsigned long long), HostMem->num_terminated_photons, 0, NULL, NULL);

	return 1;
}

void FreeMemStructs(MemStruct* HostMem, MemStruct* DeviceMem)
{
	free(HostMem->Rd_xy);
	free(HostMem->Tt_xy);
	free(HostMem->time_xyt);
	free(HostMem->tdet_pos_x);
	free(HostMem->tdet_pos_y);
	free(HostMem->fhd);
	free(HostMem->thread_active);
	free(HostMem->num_terminated_photons);

	clReleaseMemObject(DeviceMem->p);
	clReleaseMemObject(DeviceMem->Rd_xy);
	clReleaseMemObject(DeviceMem->Tt_xy);
	clReleaseMemObject(DeviceMem->time_xyt);
	clReleaseMemObject(DeviceMem->tdet_pos_x);
	clReleaseMemObject(DeviceMem->tdet_pos_y);
	clReleaseMemObject(DeviceMem->fhd);
	clReleaseMemObject(DeviceMem->bulk_info);
	clReleaseMemObject(DeviceMem->x);
  clReleaseMemObject(DeviceMem->a);
	clReleaseMemObject(DeviceMem->thread_active);
	clReleaseMemObject(DeviceMem->num_terminated_photons);

}

void FreeSimulationStruct(SimulationStruct* sim, int n_simulations)
{
	for(int i=0;i<n_simulations;i++)free(sim[i].layers);
	free(sim);
}
