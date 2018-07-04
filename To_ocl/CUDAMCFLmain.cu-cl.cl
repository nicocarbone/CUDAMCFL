/////////////////////////////////////////////////////////////
//
// Monte Carlo simulation software for light propagation in fluorescent turbid media,
// accelerated by GPU (graphic processing unit).
// The code is based on previous work by Alerstam et al and Wang et al,
// with the addition of a voxelized medium without symmetries and with an
// inhomogeneous distribution of absorbers and fluorescent marker
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
    along with CUDAMCFL.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "CUDAMCFL.h-cl.cl"
//#include "cutil.h"
//#include <float.h> //for FLT_MAX


#include "cuda_profiler_api.h-cl.cl"





















#include "CUDAMCFLio.cu-cl.cl"
#include "CUDAMCFLmem.cu-cl.cl"
#include "CUDAMCFLrng.cu-cl.cl"
#include "CUDAMCFLtransport.cu-cl.cl"

// wrapper for device code - FHD Simulation


// wrapper for device code - fluorescence Simulation



