/////////////////////////////////////////////////////////////
//
// TODO: CUDAMCFL Description
//
///////////////////////////////////////////////////////////////

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
    along with CUDAMCML_INC.  If not, see <http://www.gnu.org/licenses/>.*/

#include "CUDAMCFL.h"
#include "cutil.h"
#include <float.h> //for FLT_MAX
#include <limits.h>
#include <stdio.h>
//#include <string.h>

__device__ __constant__ unsigned long long num_photons_dc[1];
__device__ __constant__ unsigned int n_layers_dc[1];
__device__ __constant__ unsigned int start_weight_dc[1];
__device__ __constant__ LayerStruct layers_dc[MAX_LAYERS];
__device__ __constant__ DetStruct det_dc[1];
__device__ __constant__ IncStruct inclusion_dc[1];
__device__ __constant__ unsigned int ignoreAdetection_dc[1];
__device__ __constant__ unsigned int fhd_activated_dc[1];
__device__ __constant__ float xi_dc[1];
__device__ __constant__ float yi_dc[1];
__device__ __constant__ float zi_dc[1];
__device__ __constant__ float dir_dc[1];
__device__ __constant__ float esp_dc[1];

#include "CUDAMCFLio.cu"
#include "CUDAMCFLmem.cu"
#include "CUDAMCFLrng.cu"
#include "CUDAMCFLtransport.cu"

// wrapper for device code - FHD Simulation
unsigned long long DoOneSimulation(SimulationStruct *simulation, unsigned long long *x,
                     unsigned int *a, double *tempfhd) {
  MemStruct DeviceMem;
  MemStruct HostMem;
  unsigned int threads_active_total = 1;
  unsigned int i, ii;

  // Output matrix size
  const int num_x = (int)(4 * (simulation->esp) * (float)TAM_GRILLA);
  const int num_y = (int)(4 * (simulation->esp) * (float)TAM_GRILLA);
  const int num_z = (int)((simulation->esp) * (float)TAM_GRILLA);
  const int fhd_size = num_x * num_y * num_z;
  //const int fhd_size = num_x + num_x * (num_y + num_z * num_y); //x + HEIGHT* (y + WIDTH* z)

  cudaError_t cudastat;
  clock_t time1, time2;

  // Start the clock
  time1 = clock();

  // x and a are already initialised in memory
  HostMem.x = x;
  HostMem.a = a;

  InitMemStructs(&HostMem, &DeviceMem, simulation);

  InitDCMem(simulation);

  dim3 dimBlock(NUM_THREADS_PER_BLOCK);
  dim3 dimGrid(NUM_BLOCKS);

  LaunchPhoton_Global<<<dimGrid, dimBlock>>>(DeviceMem);
  CUDA_SAFE_CALL(cudaThreadSynchronize()); // Wait for all threads to finish
  cudastat = cudaGetLastError();           // Check if there was an error
  if (cudastat)
    printf("Error code=%i, %s.\n", cudastat, cudaGetErrorString(cudastat));

  i = 0;
  while (threads_active_total > 0) {
    i++;
    // run the kernel
    MCd<<<dimGrid, dimBlock>>>(DeviceMem);
    CUDA_SAFE_CALL(cudaThreadSynchronize()); // Wait for all threads to finish
    cudastat = cudaGetLastError();           // Check if there was an error
    if (cudastat)
      printf("Error code=%i, %s.\n", cudastat, cudaGetErrorString(cudastat));

    // Copy thread_active from device to host
    CUDA_SAFE_CALL(cudaMemcpy(HostMem.thread_active, DeviceMem.thread_active,
                              NUM_THREADS * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    threads_active_total = 0;
    for (ii = 0; ii < NUM_THREADS; ii++)
      threads_active_total += HostMem.thread_active[ii];

    CUDA_SAFE_CALL(cudaMemcpy(HostMem.num_terminated_photons,
                              DeviceMem.num_terminated_photons,
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (i == 50)
      printf("Estimated FHD simulation time: %.0f secs.\n",
             (double)(clock() - time1) / CLOCKS_PER_SEC *
                 (double)(simulation->number_of_photons /
                          *HostMem.num_terminated_photons));
    if (fmod(i, 200) == 0) printf("."); fflush(stdout);
    if (fmod(i, 10000) == 0)
      printf("\nRun %u, %llu photons simulated\n", i,
             *HostMem.num_terminated_photons);

  }

  CopyDeviceToHostMem(&HostMem, &DeviceMem, simulation);

  time2 = clock();

  printf("\nSimulation time: %.2f sec\n\n",
         (double)(time2 - time1) / CLOCKS_PER_SEC);

  printf("Writing excitation results...\n");
  Write_Simulation_Results(&HostMem, simulation, time2-time1);
  printf("FHD Simulation done!\n");

  unsigned long long photons_finished = *HostMem.num_terminated_photons;

  // Normalize and write output matrix
  for (int xyz = 0; xyz < fhd_size; xyz++) {
    tempfhd[xyz] = ((double)HostMem.fhd[xyz]/(0xFFFFFFFF*photons_finished));//*(double)NUMSTEPS_GPU;//((double)HostMem.fhd[xyz]+(double)FLT_MAX)*10. / photons_finished;
                   //* ((1/(float)TAM_GRILLA) * (1/(float)TAM_GRILLA) * (1/(float)TAM_GRILLA)));
  }

  printf ("Photons simulated: %llu\n\n", photons_finished);
  FreeMemStructs(&HostMem, &DeviceMem);
  return photons_finished;
}

// wrapper for device code - fluorescence Simulation
unsigned long long DoOneSimulationFl(SimulationStruct *simulation, unsigned long long *x,
                       unsigned int *a, unsigned long long *tempvoxel) {
  MemStruct DeviceMem;
  MemStruct HostMem;
  unsigned int threads_active_total = 1;
  unsigned int i, ii;

  // Size of output matrix
  const int nx2 = simulation->det.nx;
  const int ny2 = simulation->det.ny;
  const int xy_size = nx2 + ny2 * nx2;

  cudaError_t cudastat;

  // x and a are already initialised in memory
  HostMem.x = x;
  HostMem.a = a;

  InitMemStructs(&HostMem, &DeviceMem, simulation);

  InitDCMem(simulation);

  dim3 dimBlock(NUM_THREADS_PER_BLOCK);
  dim3 dimGrid(NUM_BLOCKS);

  LaunchPhoton_Global<<<dimGrid, dimBlock>>>(DeviceMem);
  CUDA_SAFE_CALL(cudaThreadSynchronize()); // Wait for all threads to finish
  cudastat = cudaGetLastError();           // Check if there was an error
  if (cudastat)
    printf("Error code=%i, %s.\n", cudastat, cudaGetErrorString(cudastat));

  i = 0;
  int watchdog = 0;
  while (threads_active_total > 0) {
    i++;
    watchdog++;
    // run the kernel
    MCd<<<dimGrid, dimBlock>>>(DeviceMem);
    CUDA_SAFE_CALL(cudaThreadSynchronize()); // Wait for all threads to finish
    cudastat = cudaGetLastError();           // Check if there was an error
    if (cudastat)
      printf("Error code=%i, %s.\n", cudastat, cudaGetErrorString(cudastat));

    // Copy thread_active from device to host
    CUDA_SAFE_CALL(cudaMemcpy(HostMem.thread_active, DeviceMem.thread_active,
                              NUM_THREADS * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    threads_active_total = 0;
    for (ii = 0; ii < NUM_THREADS; ii++)
      threads_active_total += HostMem.thread_active[ii];

    CUDA_SAFE_CALL(cudaMemcpy(HostMem.num_terminated_photons,
                              DeviceMem.num_terminated_photons,
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    if (watchdog > 10000) {
      // If we are still running after 10000 steps, something definetly went wrong.
      printf("\nWARNING: Breaking out of loop...\n");
      return 0;
    }
  }


  CopyDeviceToHostMem(&HostMem, &DeviceMem, simulation);

  for (int ijk = 0; ijk < xy_size; ijk++) {
    // Reflection
    if (simulation->do_fl_sim == 1) tempvoxel[ijk] = HostMem.Rd_xy[ijk];
    // Transmission
    if (simulation->do_fl_sim == 2) tempvoxel[ijk] = HostMem.Tt_xy[ijk];
  }

  unsigned long long photons_finished = *HostMem.num_terminated_photons;
  FreeMemStructs(&HostMem, &DeviceMem);
  return photons_finished;
}

int main(int argc, char *argv[]) {

  clock_t time0 = clock();
  SimulationStruct *simulations;
  int n_simulations;
  unsigned long long seed =
      (unsigned long long)time(NULL); // Default, use time(NULL) as seed
  int ignoreAdetection = 0;
  char *filename;
  char *filenamefl;
  //char *filenamefl3dx;
  //char *filenamefl3dy;
  //char *filenamefl3dz;
  //char *filenamefl3dzInc;
  unsigned long fhd_sim_photons;

  if (argc < 2) {
    printf("Not enough input arguments!\n");
    return 1;
  } else {
    filename = argv[1];
  }

  printf("\nExecuting %s... \n", filename);
  printf("____________________________________________________________________\n\n");

  if (interpret_arg(argc, argv, &seed, &ignoreAdetection))
    return 1;

  n_simulations =
      read_simulation_data(filename, &simulations, ignoreAdetection);

  if (n_simulations == 0) {
    printf("Something wrong with read_simulation_data!\n");
    return 1;
  } else {
    printf("Read %d simulations\n\n", n_simulations);
  }

  printf("Running FHD simulation...\n");

  // Allocate memory for RNG's
  unsigned long long x[NUM_THREADS];
  unsigned int a[NUM_THREADS];

  // Init RNG's
  if (init_RNG(x, a, NUM_THREADS, "safeprimes_base32.txt", seed))
    return 1;

  // Store in local variables the number of voxels in each direction
  const int num_x = (int)(4 * (simulations[0].esp) * (float)TAM_GRILLA);
  const int num_y = (int)(4 * (simulations[0].esp) * (float)TAM_GRILLA);
  const int num_z = (int)((simulations[0].esp) * (float)TAM_GRILLA);
  //const int fhd_size = num_x + num_x * (num_y + num_y * num_z); //x + HEIGHT* (y + WIDTH* z)
  const int fhd_size = num_x * num_y * num_z; //x + HEIGHT* (y + WIDTH* z)

  // FHD simulation
  // Perform all the simulations TODO
  // for(i=0;i<n_simulations;i++)
  //{
  // Run a simulation
  double *Fx;
  Fx = (double *)malloc((fhd_size) * sizeof(double));
  fhd_sim_photons = DoOneSimulation(&simulations[0], x, a, Fx);
  //}



  // Outputting FHD files for debug
  printf("Writing FHD files...\n"); // TODO

  // ASCII file
  FILE *fhd3DaFile_out;
  char filenamefl3da[STR_LEN];
	for (int ic=0; ic<STR_LEN; ic++) filenamefl3da[ic] = simulations[0].outp_filename[ic];
  strcat(filenamefl3da, "_FHD-Ascii.dat");

  fhd3DaFile_out = fopen(filenamefl3da, "w");
  if (fhd3DaFile_out == NULL) {
    perror("Error opening output file");
    return 0;
  }

  fprintf(fhd3DaFile_out, "%llu\t%llu\t%llu\n", num_x,num_y,num_z);

  for (int xyz = 0; xyz < fhd_size; xyz++) {
    fprintf(fhd3DaFile_out, "%.10E\n", Fx[xyz]);
  }

  fclose(fhd3DaFile_out);

  /*
  // Binary file
  FILE *fhd3DbFile_out;
  char filenamefl3db[STR_LEN];
	for (int ic=0; ic<STR_LEN; ic++) filenamefl3db[ic] = simulations[0].outp_filename[ic];
  strcat(filenamefl3db, "_FHD-Binary.dat");

  fhd3DbFile_out = fopen(filenamefl3db, "wb");
  if (fhd3DbFile_out == NULL) {
    perror("Error opening output file");
    return 0;
  }

  */

  // Fluorescence simulation

  // Initialize GPU RNG
  seed = (unsigned long long)time(NULL); // Default, use time(NULL) as seed
  if (init_RNG(x, a, NUM_THREADS, "safeprimes_base32.txt", seed))
    return 1;

  unsigned long fluor_sim_photons = 0; //Nro of simulated fluorescence photons

  if (simulations[0].do_fl_sim != 0){
    printf("Flourescence simulation... \n");
    int count_failed = 0;
    // Store in local variables the number of pixels and calculate image size
    const int nx2 = simulations[0].det.nx;
    const int ny2 = simulations[0].det.ny;
    const int xy_size = nx2 + ny2 * nx2;

    // Pixel size for Normalization
    const double dx = simulations[0].det.dx;
    const double dy = simulations[0].det.dy;

    // Initialize arrays
    double *Fl_Het;          // Final fluorescence image
    long voxel_finished = 0; // Nro of voxel simulated
    long voxel_inside = 0;   // Nro of voxel simulated inside inclusion
    long voxel_outside = 0;  // Nro of voxel simulated outside inclusion
    //const long for_size =
        num_x * num_y * num_z; // Total number of voxels to be simulated
    float xi, yi, zi;          // Temporal variable to store the voxel coordinates
    double voxelw; // Temporal variable to store the voxel scale factor
    clock_t time1,
        time2, time3; // Variable to store the timestamps used for run time stimation

    // Allocate and initialize to zero image matrix
    Fl_Het = (double *)malloc(xy_size * sizeof(double));
    for (int ijk = 0; ijk < xy_size; ijk++) {
      Fl_Het[ijk] = 0.0;
    }

    // Simulations parameters
    for (int n = 0; n < simulations[0].n_layers + 2;
        n++) { // Set mua to fluorescence value for every layer
        simulations[0].layers[n].mua = simulations[0].layers[n].muaf;
    }
    simulations[0].number_of_photons = (unsigned long long)simulations[0].number_of_photons_per_voxel; // Number of photons per voxel
    simulations[0].dir = 0.0f;        // Isotropic source
    simulations[0].fhd_activated = 0; // Don't accumulate fhd

    printf("Total fotons to be simulated: %lli over %li voxels\n",
          simulations[0].number_of_photons * fhd_size, fhd_size);

    // Loop through the voxels
    for (int ix = 0; ix < num_x; ix++) {
      for (int iy = 0; iy < num_y; iy++) {
        for (int iz = 0; iz < num_z; iz++) {
          if (ix == 0 && iy == 0 && iz == 0)
            time1 = clock(); // For the first xyz voxel, take first timestamp

          // Set source position
          xi = (ix / (float)TAM_GRILLA) - 2* simulations[0].esp;
          yi = (iy / (float)TAM_GRILLA) - 2* simulations[0].esp;
          zi = (iz / (float)TAM_GRILLA);

          simulations[0].xi = xi;
          simulations[0].yi = yi;
          simulations[0].zi = zi;

          // Locate layer of voxel (we need it to retrieve apropiate albedo)
          int found = 0;
          int nl = 1;
          while (nl < simulations[0].n_layers + 2 && found != 1) {
            if (zi < simulations[0].layers[nl].z_max &&
                zi >= simulations[0].layers[nl].z_min) {
              found = 1;
            } else
              nl++;
          }

          // Do the voxel simulation
          unsigned long long *tempret;
          tempret =
              (unsigned long long *)malloc(xy_size * sizeof(unsigned long long));
          unsigned long long voxel_status = DoOneSimulationFl(&simulations[0], x, a, tempret);

          // Check if inside inclusion and calculate scale value accordingly
          if ((xi - simulations[0].inclusion.x) *
                      (xi - simulations[0].inclusion.x) +
                  (yi - simulations[0].inclusion.y) *
                      (yi - simulations[0].inclusion.y) +
                  (zi - simulations[0].inclusion.z) *
                      (zi - simulations[0].inclusion.z) <
                simulations[0].inclusion.r * simulations[0].inclusion.r) {
          // voxel inside inclusion
            voxelw = ((double)simulations[0].inclusion.eY *
                    (double)(1 - simulations[0].inclusion.albedof) *
                    Fx[ix + num_x * (iy + iz * num_y)]) /
                    (double)(voxel_status * 0xFFFFFFFF);
            voxel_inside++;
          } else {
            // voxel ouside inclusion
            voxelw = ((double)simulations[0].layers[nl].eY *
                    (double)(1 - simulations[0].layers[nl].albedof) *
                    Fx[ix + num_x * (iy + iz * num_y)]) /
                    (double)(voxel_status * 0xFFFFFFFF);
            voxel_outside++;
          }


          if (voxel_status == 0) {
            printf("Voxel %f, %f, %f failed.\n", xi,yi,zi);
            count_failed += 1;
            }

          fluor_sim_photons += voxel_status;

          // Accumulate image
          for (int ij = 0; ij < xy_size; ij++) {
            double tempvw = voxelw * (double)tempret[ij];
            //printf ("%E\n", tempvw);
            if (Fl_Het[ij] + tempvw < DBL_MAX) Fl_Het[ij] += tempvw/(dx*dy);
					}
          voxel_finished++;
          free(tempret);
          if (fmod(voxel_finished, 200) == 0) printf("."); fflush(stdout);
          if (fmod(voxel_finished, 10000) == 0)
            printf("\n%li of %li voxels finished\n", voxel_finished, fhd_size);
          if (voxel_finished == 99) { // Second timestamp after 99 voxels run (so
                                    // it displays before the first progression
                                    // report)
          time2 = clock();
            printf("Estimated fluorescence simulation time: %.0f sec\n\n",
                 (double)(time2 - time1) * fhd_size / CLOCKS_PER_SEC / 99);
          }
        }
      }
    }

    printf("\n\nFlourescence simulation finished!\n");
    printf("Voxels inside inclusion: %li\n", voxel_inside);
    printf("Voxels outside inclusion: %li\n", voxel_outside);
    printf("Voxels failed: %i\n", count_failed);

    printf("Writing results file...\n"); // TODO
    FILE *fhdFile_out;
    filenamefl = simulations[0].outp_filename;
    strcat(filenamefl, "_Fl.dat");

    fhdFile_out = fopen(filenamefl, "w");
    if (fhdFile_out == NULL) {
      perror("Error opening output file");
      return 0;
    }

    for (int y = 0; y < ny2; y++) {
      for (int x = 0; x < nx2; x++) {
        fprintf(fhdFile_out, " %E ", Fl_Het[y * nx2 + x]);
      }
      fprintf(fhdFile_out, " \n ");
    }

    fclose(fhdFile_out);
    // Free memory
    free(Fl_Het);
    time3 = clock();
    printf("Fluorescence simulation time: %.2f sec\n\n",
         (double)(time3 - time1) /CLOCKS_PER_SEC);
  }
  free(Fx);
  FreeSimulationStruct(simulations, n_simulations);

  printf("All done! :)\n");
  printf("Total time: %.2f sec.\n", (double)(clock() - time0) /CLOCKS_PER_SEC);
  printf("Total simulated photons:\n");
  printf("\t %li FHD photons.\n", fhd_sim_photons);
  printf("\t %li Fluorescence photons.\n", fluor_sim_photons);
  printf("#############################################\n\n");
  return 0;
}
