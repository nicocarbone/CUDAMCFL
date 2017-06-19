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

// DEFINES
#define NUM_THREADS_PER_BLOCK 256//256 //128 //512 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
#define NUM_BLOCKS 128 //10  //Keep numblocks a multiple of the #MP's of the GPU (8800GT=14MP)
#define NUM_THREADS NUM_THREADS_PER_BLOCK*NUM_BLOCKS
#define NUMSTEPS_GPU 80000
#define PI 3.141592654f
#define RPI 0.318309886f
#define MAX_LAYERS 100
#define STR_LEN 500
#define MAX_STEP 14000
#define TAM_GRILLA 5
#define RAD_FIB_BAN 0.05

//#define WEIGHT 0.0001f
#define WEIGHTI 429497u //0xFFFFFFFFu*WEIGHT
#define CHANCE 0.1f


// TYPEDEFS
typedef struct __align__ (16)
{
								float z_min; // Layer z_min [cm]
								float z_max; // Layer z_max [cm]

								float mutr; // Reciprocal mu_total [cm]
								float mua; // Absorption coefficient [1/cm]
								float g; // Anisotropy factor [-]
								float n; // Refractive index [-]

								float flc; // Fluorophore concentration (Molar)
								float muaf; // Absorption coefficient at fluorescent wavelength[1/cm]
								float eY; // Energy yield	of fluorophore
								float albedof; // Albedo at emission wavelength
} LayerStruct;

typedef struct __align__ (16)
{
								float x; // Global x coordinate [cm]
								float y; // Global y coordinate [cm]
								float z; // Global z coordinate [cm]

								float dx; // (Global, normalized) x-direction
								float dy; // (Global, normalized) y-direction
								float dz; // (Global, normalized) z-direction

								unsigned int weight; // Photon weight
								int layer; // Current layer
								unsigned short bulkpos; // Current bulk descriptor
								unsigned int step; // Step actual
} PhotonStruct;

typedef struct __align__ (16)
{
								float dx; // Detection grid resolution, x-direction [cm]
								float dy; // Detection grid resolution, y-direction [cm]
								float dz; // Detection grid resolution, z-direction [cm]

								int nx; // Number of grid elements in x-direction
								int ny; // Number of grid elements in y-direction
								int nz; // Number of grid elements in z-direction TODO: why?

								float x0; // X coordinate origin of detection grid
								float y0; // X coordinate origin of detection grid

								float sep; // Separacion fibra de detaccion - eje optico TODO: remove.
} DetStruct;

typedef struct //__align__(16)
{
								float x; // Inclusion's x coordinate
								float y; // Inclusion's y coordinate
								float z; // Inclusion's z coordinate
								float r; // Inclusion's radius

								float mutr; // Mu_total reciproco de la inclusion
								float mua; // Absorption coefficient of the inclusion at excitation wavelength
								float g; // Anisotropy coefficient
								float n; // Refractive index

								float flc; // Fluorophore concentration (Molar)
								float muaf; // Absorption coefficient of the inclusion at emission wavelength
								float eY; // Energy yield
								float albedof; // Albedo at emission wavelength
}IncStruct;

typedef struct //__align__(16)
{
								float mutr; // Mu_total reciproco de la inclusion
								float mua; // Absorption coefficient of the inclusion at excitation wavelength
								float g; // Anisotropy coefficient
								float n; // Refractive index

								float flc; // Fluorophore concentration (Molar)
								float muaf; // Absorption coefficient of the inclusion at emission wavelength
								float eY; // Energy yield
								float albedof; // Albedo at emission wavelength
}BulkStruct;

typedef struct
{
								unsigned long long number_of_photons; // Number of photons to be simulated
								unsigned int number_of_photons_per_voxel; // Number of photons to be simulated per voxel
								unsigned int n_layers; // Number of layers of the medium
								unsigned int n_bulks; // Number of bulk descriptors
								unsigned int start_weight; // Photon weight at start

								char outp_filename[STR_LEN]; // Output filename
								char inp_filename[STR_LEN]; // Input filename

								long begin,end; // Input file delimitators
								char AorB; 	// Output ASCII or Binary selector

								DetStruct det; // Detector structure
								LayerStruct* layers; // Layers structure
								BulkStruct* bulks; // Bulk descriptors
								IncStruct inclusion; // Inclusion structure

								int grid_size; // Voxel size: 1cm/grid_size

								float esp; // Medium thickness

								float xi; // Source x position
								float yi; // Source y position
								float zi; // Source z position
								float dir; // Source type: 0 isotropic, 1 colimated

								//float n_up;
								float n_down;

								int fhd_activated; //0: not accumulate fhd, 1: accumulate fhd
                int do_fl_sim; //0: don't do fluorescence simulation, 1: do fluorescence simulation
								int bulk_method; // 1: single inclusion with spherical inclusion, 2: 3D described bulkd

								short* bulk_info; // 3D Matrix with a short integer per voxel selecting bulk composition

								char bulkinfo_filename[STR_LEN]; // external fila containing the bulk information for method 2

}SimulationStruct;


typedef struct
{
								PhotonStruct* p; // Pointer to structure array containing all the photon data

								unsigned long long* x; // Pointer to the array containing all the WMC x's
								unsigned int* a; // Pointer to the array containing all the WMC a's

								unsigned int* thread_active; // Pointer to the array containing the thread active status

								unsigned long long* num_terminated_photons; //Pointer to a scalar keeping track of the number of terminated photons

								unsigned long long* Rd_xy; // Matriz 2D for reflexion
								unsigned long long* Tt_xy; // Matriz 2D for transmission
								unsigned long long* fhd; // 3D Matrix for the foton hitting density, ULL because 32-bit float does not have enough precision.

								short* bulk_info; // 3D Matrix with a short integer per voxel selecting bulk composition


}MemStruct;
