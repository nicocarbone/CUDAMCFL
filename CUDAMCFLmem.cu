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

	const int num_x_tdet = sim->.det.x_temp_numdets;
  const int num_y_tdet = sim->.det.y_temp_numdets;
  const long num_tbins = sim->.det.temp_bins;
  const long timegrid_size = num_x_tdet * num_y_tdet * num_tbins;

	// Copy Rd_xy, Tt_xy and A_xyz
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->Rd_xy,DeviceMem->Rd_xy,xy_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->Tt_xy,DeviceMem->Tt_xy,xy_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	// Copy fhd
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->fhd,DeviceMem->fhd,fhd_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	// Copy time array
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->time_xyt,DeviceMem->time_xyt,timegrid_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	// Copy the state of the RNG's
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->x,DeviceMem->x,NUM_THREADS*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	return 0;
}


int InitDCMem(SimulationStruct* sim)
{
	const int num_x=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_y=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_z=(int)((sim->esp)*(double)sim->grid_size);
	const int fhd_size = num_x * num_y * num_z;

	// Copy fhd flag
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(fhd_activated_dc,&(sim->fhd_activated),sizeof(unsigned int)) );

	// Copy bulk method flag
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(bulk_method_dc,&(sim->bulk_method),sizeof(unsigned int)) );

	// Copy time sim flag
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(do_temp_sim_dc,&(sim->do_temp_sim),sizeof(unsigned int)) );


	// Copy det-data to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(det_dc,&(sim->det),sizeof(DetStruct)) );

	// Copy inclusion data to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(inclusion_dc,&(sim->inclusion),sizeof(IncStruct)) );

	// Copy number of layers to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(n_layers_dc,&(sim->n_layers),sizeof(unsigned int)));

	// Copy number of bulk descriptors to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(n_bulks_dc,&(sim->n_bulks),sizeof(unsigned int)));

	// Copy start_weight_dc to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(start_weight_dc,&(sim->start_weight),sizeof(unsigned int)));

	// Copy grid_size_dc to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(grid_size_dc,&(sim->grid_size),sizeof(unsigned int)));

	// Copy layer data to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(layers_dc,sim->layers,(sim->n_layers+2)*sizeof(LayerStruct)) );

	// Copy bulk data to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(bulks_dc,sim->bulks,(sim->n_bulks+2)*sizeof(BulkStruct)) );

	// Copy num_photons_dc to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(num_photons_dc,&(sim->number_of_photons),sizeof(unsigned long long)));

	// Copy x source position to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(xi_dc,&(sim->xi),sizeof(float)));

	// Copy y source position to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(yi_dc,&(sim->yi),sizeof(float)));

	// Copy z source position to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(zi_dc,&(sim->zi),sizeof(float)));

	// Copy source direction to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(dir_dc,&(sim->dir),sizeof(float)));

	// Copy esp to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(esp_dc,&(sim->esp),sizeof(float)));

	return 0;

}

int InitMemStructs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{
	const int xy_size = sim->det.nx + sim->det.ny*sim->det.nx; //TODO: more efficient space usage

	const int num_x=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_y=(int)(4*(sim->esp)*(double)sim->grid_size);
	const int num_z=(int)((sim->esp)*(double)sim->grid_size);
	const int fhd_size = num_x * num_y * num_z;

	const int num_x_tdet = sim->.det.x_temp_numdets;
  const int num_y_tdet = sim->.det.y_temp_numdets;
  const long num_tbins = sim->.det.temp_bins;
  const long timegrid_size = num_x_tdet * num_y_tdet * num_tbins;

	// Allocate p on the device
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->p,NUM_THREADS*sizeof(PhotonStruct)) );

	// Allocate Rd_xy on CPU and GPU
	HostMem->Rd_xy = (unsigned long long*) malloc(xy_size*sizeof(unsigned long long));
	if(HostMem->Rd_xy==NULL){printf("Error allocating HostMem->Rd_xy"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->Rd_xy,xy_size*sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->Rd_xy,0,xy_size*sizeof(unsigned long long)) );

	// Allocate Tt_xy on CPU and GPU
	HostMem->Tt_xy = (unsigned long long*) malloc(xy_size*sizeof(unsigned long long));
	if(HostMem->Tt_xy==NULL){printf("Error allocating HostMem->Tt_xy"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->Tt_xy,xy_size*sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->Tt_xy,0,xy_size*sizeof(unsigned long long)) );

	// Allocate fhd on CPU and GPU
	HostMem->fhd = (unsigned long long*) malloc(fhd_size*sizeof(unsigned long long));
	if(HostMem->fhd==NULL){printf("Error allocating HostMem->fhd"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fhd,fhd_size*sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->fhd,0,fhd_size*sizeof(unsigned long long)) );

	// Allocate timegrid on CPU and GPU
	HostMem->time_xyt = (unsigned long long*) malloc(timegrid_size*sizeof(unsigned long long));
	if(HostMem->fhd==NULL){printf("Error allocating HostMem->time_xyt"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->time_xyt,timegrid_size*sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->time_xyt,0,timegrid_size*sizeof(unsigned long long)) );

	// Allocate x time detectors
	HostMem->tdet_pos_x = (float*) malloc(num_x_tdet*sizeof(float));
	if(HostMem->tdet_pos_x==NULL){printf("Error allocating HostMem->tdet_pos_x"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->tdet_pos_x,num_x_tdet*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->tdet_pos_x,0,num_x_tdet*sizeof(float)) );

	// Allocate y time detectors
	HostMem->tdet_pos_y = (float*) malloc(num_y_tdet*sizeof(float));
	if(HostMem->tdet_pos_y==NULL){printf("Error allocating HostMem->tdet_pos_y"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->tdet_pos_y,num_y_tdet*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->tdet_pos_y,0,num_y_tdet*sizeof(float)) );

	// Allocate x and a on the device (For MWC RNG)
  CUDA_SAFE_CALL(cudaMalloc((void**)&DeviceMem->x,NUM_THREADS*sizeof(unsigned long long)));
  CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->x,HostMem->x,NUM_THREADS*sizeof(unsigned long long),cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc((void**)&DeviceMem->a,NUM_THREADS*sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->a,HostMem->a,NUM_THREADS*sizeof(unsigned int),cudaMemcpyHostToDevice));

	// Allocate bulk_info 3D matrix and copy to device memory
	CUDA_SAFE_CALL(cudaMalloc((void**)&DeviceMem->bulk_info,fhd_size*sizeof(short)));
	CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->bulk_info,sim->bulk_info,fhd_size*sizeof(short), cudaMemcpyHostToDevice ));


	// Allocate thread_active on the device and host
	HostMem->thread_active = (unsigned int*) malloc(NUM_THREADS*sizeof(unsigned int));
	if(HostMem->thread_active==NULL){printf("Error allocating HostMem->thread_active"); exit (1);}
	for(int i=0;i<NUM_THREADS;i++)HostMem->thread_active[i]=1u;
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->thread_active,NUM_THREADS*sizeof(unsigned int)) );
	CUDA_SAFE_CALL( cudaMemcpy(DeviceMem->thread_active,HostMem->thread_active,NUM_THREADS*sizeof(unsigned int),cudaMemcpyHostToDevice));

	//Allocate num_launched_photons on the device and host
	HostMem->num_terminated_photons = (unsigned long long*) malloc(sizeof(unsigned long long));
	if(HostMem->num_terminated_photons==NULL){printf("Error allocating HostMem->num_terminated_photons"); exit (1);}
	*HostMem->num_terminated_photons=0;
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->num_terminated_photons,sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemcpy(DeviceMem->num_terminated_photons,HostMem->num_terminated_photons,sizeof(unsigned long long),cudaMemcpyHostToDevice));

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

	cudaFree(DeviceMem->p);
	cudaFree(DeviceMem->Rd_xy);
	cudaFree(DeviceMem->Tt_xy);
	cudaFree(DeviceMem->time_xyt);
	cudaFree(DeviceMem->tdet_pos_x);
	cudaFree(DeviceMem->tdet_pos_y);
	cudaFree(DeviceMem->fhd);
	cudaFree(DeviceMem->bulk_info);
	cudaFree(DeviceMem->x);
  cudaFree(DeviceMem->a);
	cudaFree(DeviceMem->thread_active);
	cudaFree(DeviceMem->num_terminated_photons);

}

void FreeSimulationStruct(SimulationStruct* sim, int n_simulations)
{
	for(int i=0;i<n_simulations;i++)free(sim[i].layers);
	free(sim);
}
