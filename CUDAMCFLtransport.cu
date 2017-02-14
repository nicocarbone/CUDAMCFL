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

// forward declaration of the device code
__global__ void MCd(MemStruct);
__global__ void MCd3D(MemStruct);
__device__ float rand_MWC_oc(unsigned long long*,unsigned int*);
__device__ float rand_MWC_co(unsigned long long*,unsigned int*);
__device__ void LaunchPhoton(PhotonStruct*,unsigned long long*, unsigned int*, MemStruct);
__global__ void LaunchPhoton_Global(MemStruct);
__device__ void Spin(PhotonStruct*, float,unsigned long long*,unsigned int*);
__device__ unsigned int Reflect(PhotonStruct*, int, unsigned long long*, unsigned int*, int);
__device__ unsigned int PhotonSurvive(PhotonStruct*, unsigned long long*, unsigned int*);


__global__ void MCd(MemStruct DeviceMem)
{
  // Block index
  int bx=blockIdx.x;

  // Thread index
  int tx=threadIdx.x;

  // First element processed by the block
  int begin=NUM_THREADS_PER_BLOCK*bx;

  // Bulk thickness
	const float esp=layers_dc[(*n_layers_dc)].z_max;

  // 3D bulk matrix width and height
  const unsigned int num_x = __float2uint_rn(4*esp*__int2float_rn(*grid_size_dc));
  const unsigned int num_y = __float2uint_rn(4*esp*__int2float_rn(*grid_size_dc));

  // Size of output images
  const float size_x = __fdividef(det_dc[0].dx*__int2float_rn(det_dc[0].nx),2.);
  const float size_y = __fdividef(det_dc[0].dy*__int2float_rn(det_dc[0].ny),2.);

  unsigned long long int x=DeviceMem.x[begin+tx];//coherent
	unsigned int a=DeviceMem.a[begin+tx];//coherent

	float s;	// step length
	unsigned int index; // temporal variable to store indexes to arrays
	unsigned int w; // photon weight

	PhotonStruct p = DeviceMem.p[begin+tx];

	int new_layer;

	//First, make sure the thread (photon) is active
	unsigned int ii = 0;
	if(!DeviceMem.thread_active[begin+tx]) ii = NUMSTEPS_GPU;

	for(;ii<NUMSTEPS_GPU;ii++) {
    //this is the main while loop

    // Check if inside spherical inclusion
    if((((p.x-inclusion_dc[0].x)*(p.x-inclusion_dc[0].x)) +
       ((p.y-inclusion_dc[0].y)*(p.y-inclusion_dc[0].y)) +
       ((p.z-inclusion_dc[0].z)*(p.z-inclusion_dc[0].z))) <=
       (inclusion_dc[0].r*inclusion_dc[0].r)) {
      // Inside inclusion
			if(inclusion_dc[0].mutr!=FLT_MAX)
				s = -__logf(rand_MWC_oc(&x,&a))*inclusion_dc[0].mutr;//sample step length [cm] //HERE AN OPEN_OPEN FUNCTION WOULD BE APPRECIATED
			else
				s = 2.0f;//temporary, say the step in glass is 100 cm.
		}
		else 	{
      // Outside inclusion
			if(layers_dc[p.layer].mutr!=FLT_MAX)
				s = -__logf(rand_MWC_oc(&x,&a))*layers_dc[p.layer].mutr;//sample step length [cm] //HERE AN OPEN_OPEN FUNCTION WOULD BE APPRECIATED
			else
				s = 2.0f;//temporary, say the step in glass is 100 cm.
		}

		//Check for layer transitions and in case, calculate s
		new_layer = p.layer;

    //Check for upwards reflection/transmission & calculate new s
		if(p.z+s*p.dz<layers_dc[p.layer].z_min) {
      new_layer--;
      s = __fdividef(layers_dc[p.layer].z_min-p.z,p.dz);
      }

    //Check for downward reflection/transmission
		if(p.z+s*p.dz>=layers_dc[p.layer].z_max){
      new_layer++;
      s = __fdividef(layers_dc[p.layer].z_max-p.z,p.dz);
    }

    //Move photon
    p.x += p.dx*s;
		p.y += p.dy*s;
		p.z += p.dz*s;

    if(p.z>layers_dc[p.layer].z_max)p.z=layers_dc[p.layer].z_max;//needed? TODO
		if(p.z<layers_dc[p.layer].z_min)p.z=layers_dc[p.layer].z_min;//needed? TODO

    // Accumulate photon hitting density if needed
    if (*fhd_activated_dc == 1){
      if(fabsf(p.x)<2*esp && fabsf(p.y)<2*esp && p.z<esp){ //Inside space of fhd
        //Use round to zero so there are no over sampled voxels (for ex: (max_x,0,0) and (0,1,0) should not map to the same voxel)
        index = __float2uint_rz((p.x+2*esp)*__int2float_rn(*grid_size_dc))
                  + num_x * (__float2uint_rz((p.y+2*esp)*__int2float_rn(*grid_size_dc))
                  + num_y * __float2uint_rz((p.z)*__int2float_rn(*grid_size_dc))); //x + HEIGHT* (y + WIDTH* z) Fx[ix + num_x * (iy + iz * num_y)]

        if (DeviceMem.fhd[index] + p.weight < LLONG_MAX) atomicAdd(&DeviceMem.fhd[index], p.weight); // Check for overflow and add atomically //TODO why LLONG_MAX?
      }
    }

    // New layer?
		if(new_layer!=p.layer) {
      // set the remaining step length to 0
			s = 0.0f;

      if(Reflect(&p,new_layer,&x,&a,1)==0u) {
        // Photon is transmitted

        if(new_layer == 0){
          // Diffuse reflectance
          if((fabsf(p.x)-det_dc[0].x0)<size_x && (fabsf(p.y)-det_dc[0].y0)<size_y) {
            // Photon is detectable
            // Use round to zero so there are no over sampled pixels (for ex: (max_x,0) and (0,1) should not map to the same pixel)
            index=__float2uint_rz(__fdividef(p.y-det_dc[0].y0+size_y,det_dc[0].dy)) * det_dc[0].nx +
                  __float2uint_rz(__fdividef(p.x-det_dc[0].x0+size_x,det_dc[0].dx));
    				if ((DeviceMem.Rd_xy[index] + p.weight) < LLONG_MAX) atomicAdd(&DeviceMem.Rd_xy[index], p.weight); // Check for overflow and add atomicall
    			}
          p.weight = 0; // Set the remaining weight to 0, effectively killing the photon
        }

        if(new_layer > *n_layers_dc) {
          // Diffuse transmitance
          if((fabsf(p.x)-det_dc[0].x0)<size_x && (fabsf(p.y)-det_dc[0].y0)<size_y) {
            // Photon transmitted
            // Use round to zero so there are no over sampled pixels (for ex: (max_x,0) and (0,1) should not map to the same pixel)
            index=__float2uint_rz(__fdividef(p.y-det_dc[0].y0+size_y,det_dc[0].dy)) * det_dc[0].nx +
                  __float2uint_rz(__fdividef(p.x-det_dc[0].x0+size_x,det_dc[0].dx));
            if ((DeviceMem.Tt_xy[index] + p.weight) < LLONG_MAX) atomicAdd(&DeviceMem.Tt_xy[index], p.weight); // Check for overflow and add atomically
          }
					p.weight = 0; // Set the remaining weight to 0, killing the photon
        }
			}
		}

		w=0;

		if(s > 0.0f) {
			// Drop weight (apparently only when the photon is scattered)
			if((((p.x-inclusion_dc[0].x)*(p.x-inclusion_dc[0].x)) +
          ((p.y-inclusion_dc[0].y)*(p.y-inclusion_dc[0].y)) +
          ((p.z-inclusion_dc[0].z)*(p.z-inclusion_dc[0].z))) <=
          (inclusion_dc[0].r*inclusion_dc[0].r)) {
            // Inside inclusion
				    w = __float2uint_rn(inclusion_dc[0].mua*inclusion_dc[0].mutr*__uint2float_rn(p.weight));
            if (p.weight - w >= 0) // Check for underflow
              p.weight -= w;
            else
              p.weight = 0;
				    Spin(&p,inclusion_dc[0].g,&x,&a);
      }
			else {
        // Outside inclusion
				w = __float2uint_rn(layers_dc[p.layer].mua*layers_dc[p.layer].mutr*__uint2float_rn(p.weight));
        if (p.weight - w >= 0) // Check for underflow
          p.weight -= w;
        else
          p.weight = 0;
				Spin(&p,layers_dc[p.layer].g,&x,&a);
			}
		}

		if(!PhotonSurvive(&p,&x,&a)) // Check if photons survives or not
		{
			if(atomicAdd(DeviceMem.num_terminated_photons,1ULL) < (*num_photons_dc-NUM_THREADS)) {
        // Ok to launch another photon
				LaunchPhoton(&p,&x,&a, DeviceMem);//Launch a new photon
      }
			else {
        // No more photons should be launched.
				DeviceMem.thread_active[begin+tx] = 0u; // Set thread to inactive
				ii = NUMSTEPS_GPU;				// Exit main loop
			}

		}
	}//end main for loop!
	__syncthreads();//necessary?

	//save the state of the MC simulation in global memory before exiting
	DeviceMem.p[begin+tx] = p;	//This one is incoherent!!!
	DeviceMem.x[begin+tx] = x; //this one also seems to be coherent

}//end MCd


__global__ void MCd3D(MemStruct DeviceMem)
{

  // Block index
  int bx=blockIdx.x;

  // Thread index
  int tx=threadIdx.x;

  // First element processed by the block
  int begin=NUM_THREADS_PER_BLOCK*bx;

  // 3D bulk matrix width and height
  const unsigned int num_x = __float2uint_rn(4*(*esp_dc)*__int2float_rn(*grid_size_dc));
  const unsigned int num_y = __float2uint_rn(4*(*esp_dc)*__int2float_rn(*grid_size_dc));

  // Size of output images
  const float size_x = __fdividef(det_dc[0].dx*__int2float_rn(det_dc[0].nx),2.);
  const float size_y = __fdividef(det_dc[0].dy*__int2float_rn(det_dc[0].ny),2.);

  // Last Bulk
  const unsigned short last_bulk = *n_bulks_dc+1;

  unsigned long long int x=DeviceMem.x[begin+tx];//coherent
	unsigned int a=DeviceMem.a[begin+tx];//coherent

	float s;	// step length
	unsigned int index; // temporal variable to store indexes to arrays
	unsigned int w; // photon weight

	PhotonStruct p = DeviceMem.p[begin+tx];

	unsigned int new_bulk; // int storing bulk descriptor of current position

	// First, make sure the thread (photon) is active
	unsigned int ii = 0;
	if(!DeviceMem.thread_active[begin+tx]) ii = NUMSTEPS_GPU;

	for(;ii<NUMSTEPS_GPU;ii++) {
    // Main while loop

  	if(bulks_dc[p.bulkpos].mutr!=FLT_MAX)
			s = -__logf(rand_MWC_oc(&x,&a))*bulks_dc[p.bulkpos].mutr;//sample step length [cm] //HERE AN OPEN_OPEN FUNCTION WOULD BE APPRECIATED
		else
			//s = 100.0f; //temporary, say the step in glass is 100 cm.
      s = 2.0f;

		//Check for layer transitions and in case, calculate s
		new_bulk = p.bulkpos;

    //int side_scape = 0;

    //Check for upwards reflection/transmission and move to surface
		if(p.z+s*p.dz<0.) {
      new_bulk = 0;
      //side_scape = 0;
      s = __fdividef(-p.z,p.dz);
    }

    //Check for downward reflection/transmission and move to surface
		if(p.z+s*p.dz>(*esp_dc)){
      new_bulk = last_bulk;
      //side_scape = 0;
      s = __fdividef((*esp_dc)-p.z,p.dz);
    }

    // Move photon TODO:here?
    p.x += p.dx*s;
    p.y += p.dy*s;
    p.z += p.dz*s;

    //if(p.z>(*esp_dc)) p.z=(*esp_dc);//needed? TODO
		//if(p.z<0.) p.z=0.;//needed? TODO

    // Retrieve bulk position
    if(new_bulk!=0 && new_bulk!=last_bulk) {
      if (fabsf(p.x)<2*(*esp_dc) && fabsf(p.y)<2*(*esp_dc) && p.z<(*esp_dc)){
        // Inside space of 3D matrix. Calculate index of the current voxel
        // Use round to zero so there are no over sampled voxels (for ex: (max_x,0,0) and (0,1,0) should not map to the same voxel)
        index = __float2uint_rz((p.x+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
              + num_x * (__float2uint_rz((p.y+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
              + num_y * __float2uint_rz((p.z)*__int2float_rn(*grid_size_dc)));
        new_bulk = DeviceMem.bulk_info[index];
      }
      else {
        // Photon scaped to the sides
        // Outside space of 3D matrix, assume inside homogeneous medium (should we assume it is outside the bulk (bulkpos=0)? TODO)
        new_bulk = 1;
        }
    }

    unsigned int reflected;

		if(new_bulk != p.bulkpos) {
      // If changing voxel do Reflect
      reflected = Reflect(&p,new_bulk,&x,&a,2);
    }

    // Accumulate photon hitting density if needed
    if(fabsf(p.x)<2*(*esp_dc) && fabsf(p.y)<2*(*esp_dc) && p.z<(*esp_dc) && *fhd_activated_dc == 1) {
      // Inside space of 3D matrix. Calculate index of the current voxel
      // Use round to zero so there are no over sampled voxels (for ex: (max_x,0,0) and (0,1,0) should not map to the same voxel)
      index = __float2uint_rz((p.x+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
            + num_x * (__float2uint_rz((p.y+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
            + num_y * __float2uint_rz((p.z)*__int2float_rn(*grid_size_dc)));
      if (DeviceMem.fhd[index] + p.weight < LLONG_MAX) atomicAdd(&DeviceMem.fhd[index], p.weight); // Check for overflow and add atomically
      //TODO why LLONG_MAX?
    }


    if(p.bulkpos == 0){
      // Photon is outside bulk and reflected
      if((fabsf(p.x)-det_dc[0].x0)<size_x && (fabsf(p.y)-det_dc[0].y0)<size_y) {
        // Photon is detectable
        // Use round to zero so there are no over sampled pixels (for ex: (max_x,0) and (0,1) should not map to the same pixel)
        index=__float2uint_rz(__fdividef(p.y-det_dc[0].y0+size_y,det_dc[0].dy)) * det_dc[0].nx +
              __float2uint_rz(__fdividef(p.x-det_dc[0].x0+size_x,det_dc[0].dx));
				if ((DeviceMem.Rd_xy[index] + p.weight) < LLONG_MAX) atomicAdd(&DeviceMem.Rd_xy[index], p.weight); // Check for overflow and add atomicall
				}
      p.weight = 0u; // Set the remaining weight to 0, effectively killing the photon
      s = 0.0f;
    }

    if(p.bulkpos == last_bulk){
      // Photon is outside bulk and transmitted
      if((fabsf(p.x)-det_dc[0].x0)<size_x && (fabsf(p.y)-det_dc[0].y0)<size_y) {
        // Photon transmitted
        // Use round to zero so there are no over sampled pixels (for ex: (max_x,0) and (0,1) should not map to the same pixel)
        index=__float2uint_rz(__fdividef(p.y-det_dc[0].y0+size_y,det_dc[0].dy)) * det_dc[0].nx +
              __float2uint_rz(__fdividef(p.x-det_dc[0].x0+size_x,det_dc[0].dx));
        if ((DeviceMem.Tt_xy[index] + p.weight) < LLONG_MAX) atomicAdd(&DeviceMem.Tt_xy[index], p.weight); // Check for overflow and add atomically
        }
      p.weight = 0u; // Set the remaining weight to 0, effectively killing the photon
      s = 0.0f;
      }

		w=0;

		if(s > 0.0f) {
    	// Drop weight (apparently only when the photon is scattered)
			w = __float2uint_ru(bulks_dc[p.bulkpos].mua*bulks_dc[p.bulkpos].mutr*__uint2float_rn(p.weight));
			if (p.weight - w >= 0 && w > 0) // Check for underflow
        p.weight -= w;
      else
        p.weight = 0u;
			Spin(&p,bulks_dc[p.bulkpos].g,&x,&a);
		}

		if(!PhotonSurvive(&p,&x,&a)) {
      // Check if photons survives or not
			if(atomicAdd(DeviceMem.num_terminated_photons,1ULL) < (*num_photons_dc-NUM_THREADS)) {
        // Ok to launch another photon
				LaunchPhoton(&p,&x,&a, DeviceMem);//Launch a new photon
      }
			else {
        // No more photons should be launched.
				DeviceMem.thread_active[begin+tx] = 0u; // Set thread to inactive
				ii = NUMSTEPS_GPU;			               	// Exit main loop
			}

		}

	}//end main for loop!
	__syncthreads();//necessary?

	//save the state of the MC simulation in global memory before exiting
	DeviceMem.p[begin+tx] = p;	//This one is incoherent!!!
	DeviceMem.x[begin+tx] = x; //this one also seems to be coherent

}//end MCd3D



__device__ void LaunchPhoton(PhotonStruct* p, unsigned long long* x, unsigned int* a, MemStruct DeviceMem)
{

  // 3D bulk matrix width and height
  const unsigned int num_x = __float2uint_rn(4*(*esp_dc)*__int2float_rn(*grid_size_dc));
  const unsigned int num_y = __float2uint_rn(4*(*esp_dc)*__int2float_rn(*grid_size_dc));

  if (*dir_dc == 0.0f) {
    // Isotropic source, random position in voxel size
    // We are using round to zero in PHD, so pixels are mapped to 0 boundary
    p->x  = *xi_dc + (0.999999f/__int2float_rn(*grid_size_dc))*rand_MWC_oc(x,a);
  	p->y  = *yi_dc + (0.999999f/__int2float_rn(*grid_size_dc))*rand_MWC_oc(x,a);
    p->z  = *zi_dc + (0.999999f/__int2float_rn(*grid_size_dc))*rand_MWC_oc(x,a);

    float costheta = 1.0 - 2.0*rand_MWC_oc(x,a);
	  float sintheta = sqrt(1.0 - costheta*costheta);
	  float psi = 2.0*PI*rand_MWC_oc(x,a);
    float cospsi = __cosf(psi);
    float sinpsi;

    if (psi < PI)
		  sinpsi = __fsqrt_rn(1.0 - cospsi*cospsi);
	  else
		  sinpsi = -__fsqrt_rn(1.0 - cospsi*cospsi);

    p->dx = sintheta*cospsi;
	  p->dy = sintheta*sinpsi;
	  p->dz = costheta;

    p->weight = 0xFFFFFFFF; // no specular reflection (Initial weight: max int32)
    if ((*bulk_method_dc) == 2){
      if (p->z<(*esp_dc)){
        if(fabsf(p->x)<2*(*esp_dc) && fabsf(p->y)<2*(*esp_dc)){ //Inside space of fhd
          //Use round to zero so there are no over sampled voxels (for ex: (max_x,0,0) and (0,1,0) should not map to the same voxel)
          unsigned int index = __float2uint_rz((p->x+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
                              + num_x * (__float2uint_rz((p->y+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
                              + num_y * __float2uint_rz((p->z)*__int2float_rn(*grid_size_dc)));
          p->bulkpos = DeviceMem.bulk_info[index];
        }
        else p->bulkpos = 1;
      }
      else p->bulkpos = *n_bulks_dc+1;
    }
    else p->bulkpos = 0;

  }
  else {
    // Colimated source
    // We need to randomize the foton injection even for a colimated source. Otherwize, the FHD will have some bias to a pariticular voxel if
    // the source hits the boundary between voxels.
    const float input_fibre_diameter = 0.03;//[cm]

    float sample_rad = input_fibre_diameter*__fsqrt_rn(-__logf(rand_MWC_oc(x,a)));
    float sample_phi = 2*PI*rand_MWC_oc(x,a);

    float sin_phi;
    float cos_phi;
    __sincosf (sample_phi, &sin_phi, &cos_phi);

    p->x = *xi_dc + sample_rad * cos_phi;
    p->y = *yi_dc + sample_rad * sin_phi;
    p->z = *zi_dc;

    p->dx = 0.0f;
	  p->dy = 0.0f;
	  p->dz = 1.0f;

    p->weight = *start_weight_dc; // specular reflection at boundary

    if ((*bulk_method_dc) == 2){
      if (p->z<(*esp_dc)){
        if(fabsf(p->x)<2*(*esp_dc) && fabsf(p->y)<2*(*esp_dc)){ //Inside space of fhd
          //Use round to zero so there are no over sampled voxels (for ex: (max_x,0,0) and (0,1,0) should not map to the same voxel)
          unsigned int index = __float2uint_rz((p->x+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
                              + num_x * (__float2uint_rz((p->y+2*(*esp_dc))*__int2float_rn(*grid_size_dc))
                              + num_y * __float2uint_rz((p->z)*__int2float_rn(*grid_size_dc)));
          p->bulkpos = DeviceMem.bulk_info[index];
        }
        else p->bulkpos = 1;
      }
      else p->bulkpos = *n_bulks_dc+1;
    }
    else p->bulkpos = 0;

  }

	p->step= 0;

  if ((*bulk_method_dc) == 1){
  // Found photon start layer
    int found = 0;
    int nl = 1;
    while (nl < *n_layers_dc+2 && found != 1) {
      if (*zi_dc < layers_dc[nl].z_max && *zi_dc >= layers_dc[nl].z_min){
        p->layer = nl;
        found = 1;
      }
      else nl++;
    }
  }
  else p->layer = 0;
}


__global__ void LaunchPhoton_Global(MemStruct DeviceMem)//PhotonStruct* pd, unsigned long long* x, unsigned int* a)
{
	int bx=blockIdx.x;
  int tx=threadIdx.x;

  //First element processed by the block
  int begin=NUM_THREADS_PER_BLOCK*bx;

	PhotonStruct p;
	unsigned long long int x=DeviceMem.x[begin+tx];//coherent

	unsigned int a=DeviceMem.a[begin+tx];//coherent

	LaunchPhoton(&p,&x,&a, DeviceMem);

	//__syncthreads();//necessary?
	DeviceMem.p[begin+tx]=p;//incoherent!?
}


__device__ void Spin(PhotonStruct* p, float g, unsigned long long* x, unsigned int* a)
{
	float cost, sint;	// cosine and sine of the
						// polar deflection angle theta.
	float cosp, sinp;	// cosine and sine of the
						// azimuthal angle psi.
	float temp;

	float tempdir=p->dx;

	// This is more efficient for g!=0 but of course less efficient for g==0
	temp = __fdividef((1.0f-(g)*(g)),(1.0f-(g)+2.0f*(g)*rand_MWC_co(x,a)));// Should be close close????!!!!!
	cost = __fdividef((1.0f+(g)*(g) - temp*temp),(2.0f*(g)));
	if(g==0.0f)
		cost = 2.0f*rand_MWC_co(x,a) -1.0f;// Should be close close??!!!!!

	sint = sqrtf(1.0f - cost*cost);

	__sincosf(2.0f*PI*rand_MWC_co(x,a),&cosp,&sinp);// spin psi [0-2*PI)

	temp = sqrtf(1.0f - p->dz*p->dz);

	if(temp==0.0f) // Normal incident.
	{
		p->dx = sint*cosp;
		p->dy = sint*sinp;
		p->dz = copysignf(cost,p->dz*cost);
	}
	else // Regular incident.
	{
		p->dx = __fdividef(sint*(p->dx*p->dz*cosp - p->dy*sinp),temp) + p->dx*cost;
		p->dy = __fdividef(sint*(p->dy*p->dz*cosp + tempdir*sinp),temp) + p->dy*cost;
		p->dz = -sint*cosp*temp + p->dz*cost;
	}

	// Normalisation seems to be required as we are using floats! Otherwise the small numerical error will accumulate
	temp=rsqrtf(p->dx*p->dx+p->dy*p->dy+p->dz*p->dz);
	p->dx = p->dx*temp;
	p->dy = p->dy*temp;
	p->dz = p->dz*temp;
}// end Spin


__device__ unsigned int Reflect(PhotonStruct* p, int newlb, unsigned long long* x, unsigned int* a, int bulk_method)
{
	// Calculates whether the photon is reflected (returns 1) or not (returns 0)
	// Reflect() will also update the current photon layer (after transmission) and photon direction (both transmission and reflection)
  float n1, n2;
  if (bulk_method==1){
	  n1 = layers_dc[p->layer].n;
	  n2 = layers_dc[newlb].n;
  }
  else if (bulk_method==2){
    n1 = bulks_dc[p->bulkpos].n;
    n2 = bulks_dc[newlb].n;
  }
	float r;
	float cos_angle_i = fabsf(p->dz);

	if(n1==n2)//refraction index matching automatic transmission and no direction change
	{
		p->layer = newlb;
    p->bulkpos = newlb;
		return 0u;
	}

	if(n2*n2<n1*n1*(1-cos_angle_i*cos_angle_i))//total internal reflection, no layer change but z-direction mirroring
	{
		p->dz *= -1.0f;
		return 1u;
	}

	if(cos_angle_i==1.0f)//normal incident
	{
		r = __fdividef((n1-n2),(n1+n2));
		if(rand_MWC_co(x,a)<=r*r)
		{
			//reflection, no layer change but z-direction mirroring
			p->dz *= -1.0f;
			return 1u;
		}
		else
		{	//transmission, no direction change but layer change
			p->layer = newlb;
      p->bulkpos = newlb;
			return 0u;
		}
	}

	// gives almost exactly the same results as the old MCML way of doing the calculation but does it slightly faster
	// save a few multiplications, calculate cos_angle_i^2;
	float e = __fdividef(n1*n1,n2*n2)*(1.0f-cos_angle_i*cos_angle_i); //e is the sin square of the transmission angle
	r=2*sqrtf((1.0f-cos_angle_i*cos_angle_i)*(1.0f-e)*e*cos_angle_i*cos_angle_i);//use r as a temporary variable
	e=e+(cos_angle_i*cos_angle_i)*(1.0f-2.0f*e);//Update the value of e
	r = e*__fdividef((1.0f-e-r),((1.0f-e+r)*(e+r)));//Calculate r

	if(rand_MWC_co(x,a)<=r)
	{
		// Reflection, mirror z-direction!
		p->dz *= -1.0f;
		return 1u;
	}
	else
	{
		// Transmission, update layer/descritor and direction
		r = __fdividef(n1,n2);
		e = r*r*(1.0f-cos_angle_i*cos_angle_i); //e is the sin square of the transmission angle
		p->dx *= r;
		p->dy *= r;
		p->dz = copysignf(sqrtf(1-e) ,p->dz);
		p->layer = newlb;
    p->bulkpos = newlb;
		return 0u;
	}

}


__device__ unsigned int PhotonSurvive(PhotonStruct* p, unsigned long long* x, unsigned int* a)
{	//Calculate wether the photon survives (returns 1) or dies (returns 0)

	if(p->weight>WEIGHTI) return 1u; // No roulette needed
	if(p->weight==0u) return 0u;	// Photon has exited slab, i.e. kill the photon

	if(rand_MWC_co(x,a)<CHANCE)
	{
		p->weight = __float2uint_rn(__fdividef(__uint2float_rn(p->weight),CHANCE));
    return 1u;
  }

	//else
	return 0u;
}

/*
//Device function to add an unsigned integer to an unsigned long long using CUDA. Needed for Compute Capability 1.1
__device__ void AtomicAddULL(unsigned long long* address, unsigned int add)
{
	if(atomicAdd((unsigned int*)address,add)+add<add)
		atomicAdd(((unsigned int*)address)+1,1u);
}
*/
