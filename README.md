# CUDAMCFL

Monte Carlo simulation software for light propagation in fluorescent turbid media, accelerated by GPU (graphic processing unit). The code is based on previous work by Alerstam et al and Wang et al, with the addition of a voxelized medium without symmetries and with an inhomogeneous distribution of absorbers and fluorescent markers.

CUDAMCFLmain.cu
-------------------
Main routine. 

CUDAMCFLtransport.cu
-------------------
Light launch and transport routine.

CUDAMCFLmem.cu
-------------------
Memory management, allocation and copy between host and device.

CUDAMCFLio.cu
-------------------
Input/output routines for reading simulation properties and writting results.

CUDAMCFLrng.cu
-------------------
Random number generator routines implemented in CUDA

CUDAMDFL.h
-------------------
Routines headers.

Compile
-------------------
Bash script for compiling. General command.

Compile750
-------------------
Bash script for compiling. Tailored for Nvidia GTX750Ti.

safeprimes_base32.txt
-------------------
List of safe primes for the random number generator.

Bulk_descriptor_creator.ipynb
-------------------
iPython script for the generation of bulk descriptor matrices. Provided as an example, it can create media with spherical or cylndrical embedded inhomogeneities.

Bulk_descriptor_visualizator.ipynb
-------------------
iPython script for the visualization of bulk descriptor matrices.

sample.mci
-------------------
Sample MCI file defining the inclusion as a sphere without the use of a bulk descriptor input matrix.

sample2.mci
-------------------
Sample MCI file defining using a bulk descriptor input matrix.


