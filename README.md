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


## Research paper

This code is discussed in a peer-reviewed paper published in Biomedical Physics & Engineering Express under the title: "GPU accelerated Monte Carlo simulation of light propagation in inhomogeneous fluorescent turbid media. Application to whole field CW imaging.".

If you are using CUDAMCFL for academic purposes, please cite the preceding paper. In BibTeX form:

```latex
@article{10.1088/2057-1976/aa7b8f,
  author={Nicolás Abel Carbone and Juan A Pomarico and Daniela Inés Iriarte},
    title={GPU accelerated Monte Carlo simulation of light propagation in inhomogeneous fluorescent turbid media. Application to whole field CW imaging.},
  journal={Biomedical Physics & Engineering Express},
        
  url={http://iopscience.iop.org/10.1088/2057-1976/aa7b8f},
  year={2017},
  abstract={Abstract We present an implementation of a Monte Carlo simulation software for fluorescent turbid media, accelerated by GPU (Graphic Processing Unit). &#13; The code is based on previous work by Alerstam et al. and Wang et al., with the addition of a voxelized medium without symmetries and with an inhomogeneous distribution of absorbers and fluorescent markers. &#13; Cartesian coordinates are used in place of the cylindrical ones used in previous versions. &#13; It is particularly aimed at the simulation of CW whole-field reflectance and transmittance images of fluorescence and absorption.&#13; Several tests and comparisons with numerical and theoretical techniques were performed in order to validate our approach.}
```
