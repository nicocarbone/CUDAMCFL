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



