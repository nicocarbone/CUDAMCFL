/*	This file is part of CUDAMCML_INC.

    CUDAMCML_INC is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCML_INC is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCML_INC.  If not, see <http://www.gnu.org/licenses/>.*/

#define NFLOATS 12
#define NINTS 5

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

int interpret_arg(int argc, char* argv[], unsigned long long* seed, int* ignoreAdetection)
{

	int unknown_argument;
	for(int i=2;i<argc;i++)
	{
		unknown_argument=1;
		if(!strcmp(argv[i],"-A"))
		{
			unknown_argument=0;
			*ignoreAdetection=1; //This option is not yet implemented. Therefore, this option has no effect.
			printf("Ignoring A-detection!\n");
		}
		if(!strncmp(argv[i],"-S",2) && sscanf(argv[i],"%*2c %llu",seed))
		{
		unknown_argument=0;
		printf("Seed=%llu\n",*seed);
		}
		if(unknown_argument)
		{
			printf("Unknown argument %s!\n",argv[i]);
			return 1;
		}
	}
	return 0;
}

int Write_Simulation_Results(MemStruct* HostMem, SimulationStruct* sim, clock_t simulation_time)
{
	FILE* pFile_inp;
	//FILE* pFile_outp;
	char mystring[STR_LEN];

	// Copy stuff from sim->det to make things more readable:
	double dx=(double)sim->det.dx;		// Detection grid resolution, x-direction [cm]
	double dy=(double)sim->det.dy;		// Detection grid resolution, y-direction [cm]

	int nx=sim->det.nx;			// Number of grid elements in x-direction
	int ny=sim->det.ny;			// Number of grid elements in y-direction


	int x,y;//,z;
	//unsigned int l;
	//int i;

	//unsigned long long temp=0;
	double scale1 = (double)0xFFFFFFFF*(double)sim->number_of_photons; // Number of photons (used to normalize)
	double scale2;

	// Open the input and output files
	pFile_inp = fopen (sim->inp_filename , "r");
	if (pFile_inp == NULL){perror ("Error opening input file");return 0;}

	FILE *paramFile_out;
	char filenameparam[STR_LEN];
	for (int ic=0; ic<STR_LEN; ic++) filenameparam[ic] = sim->outp_filename[ic];
	strcat(filenameparam, "_param.dat");

	paramFile_out = fopen (filenameparam , "w");
	if (paramFile_out == NULL){perror ("Error opening parameters output file");return 0;}


	// Write other stuff here first!

	fprintf(paramFile_out,"A1 	# Version number of the file format.\n\n");
	fprintf(paramFile_out,"####\n");
	fprintf(paramFile_out,"# Data categories include: \n");
	fprintf(paramFile_out,"# InParm, RAT, \n");
	fprintf(paramFile_out,"# A_l, A_z, Rd_r, Rd_a, Tt_r, Tt_a, \n");
	fprintf(paramFile_out,"# A_rz, Rd_ra, Tt_ra \n");
	fprintf(paramFile_out,"####\n\n");

	// Write simulation time
	fprintf(paramFile_out,"# User time: %.2f sec\n\n",(double)simulation_time/CLOCKS_PER_SEC);

	fprintf(paramFile_out,"# TAM_GRILLA: %i\n\n",TAM_GRILLA);

	fprintf(paramFile_out,"InParam\t\t# Input parameters:\n");
	// Copy the input data from inp_filename
	fseek(pFile_inp, sim->begin, SEEK_SET);
	while(sim->end>ftell(pFile_inp))
	{

		fgets(mystring , STR_LEN , pFile_inp);
		fputs(mystring , paramFile_out);
	}


	fclose(pFile_inp);
	fclose(paramFile_out);


	FILE *transFile_out;
	char filenametrans[STR_LEN];
	for (int ic=0; ic<STR_LEN; ic++) filenametrans[ic] = sim->outp_filename[ic];
	strcat(filenametrans, "_Trans.dat");
	transFile_out = fopen (filenametrans , "w");
	if (transFile_out == NULL){perror ("Error opening transmission output file");return 0;}
	//int i=0;
	//fprintf(pFile_outp,"\n\n# T[x][y]. [1/(cm2)].\n# T[0][0], [0][1],..[0][ny-1]\n# T[1][0], [1][1],..[1][ny-1]\n# ...\n# T[nx-1][0], [nx-1][1],..[nx-1][ny-1]\nT\n");
	for(x=0;x<nx;x++)
	{
		for(y=0;y<ny;y++)
		{
			scale2=scale1*dx*dy; // Normalization Constant
			fprintf(transFile_out," %E ",(double)HostMem->Tt_xy[x*ny+y]/scale2);
		}
		fprintf(transFile_out," \n ");
	}

	fclose(transFile_out);

	FILE *reflFile_out;
	char filenamerefl[STR_LEN];
	for (int ic=0; ic<STR_LEN; ic++) filenamerefl[ic] = sim->outp_filename[ic];
	strcat(filenamerefl, "_Refl.dat");
	reflFile_out = fopen (filenamerefl , "w");
	if (reflFile_out == NULL){perror ("Error opening reflection output file");return 0;}
	//i=0;
	//fprintf(relfFile_out,"\n\n# R[x][y]. [1/(cm2)].\n# R[0][0], [0][1],..[0][ny-1]\n# R[1][0], [1][1],..[1][ny-1]\n# ...\n# R[nx-1][0], [nx-1][1],..[nx-1][ny-1]\nR\n");
	for(x=0;x<nx;x++)
	{
		for(y=0;y<ny;y++)
		{
			scale2=scale1*dx*dy; // Normalization Constant
			fprintf(reflFile_out," %E ",(double)HostMem->Rd_xy[x*ny+y]/scale2);
		}
		fprintf(reflFile_out," \n ");
	}


	fclose(reflFile_out);

/*
	FILE* bFile_out;
	bFile_out = fopen ("salida_fhd.dat","w");
	int num_z=(int)(sim->esp)*TAM_GRILLA;
	int num_x=(int)2*(sim->esp)*TAM_GRILLA;
	int num_y=(int)2*(sim->esp)*TAM_GRILLA;
	float xf,yf,zf;
	for(int ix=0;ix<num_x;ix++){
		for(int iy=0;iy<num_y;iy++){
        	for(int iz=0;iz<num_z;iz++){
								xf = (ix/num_x)*2*(sim->esp) - (sim->esp);
								yf = (iy/num_y)*2*(sim->esp) - (sim->esp);
								zf = (iz/num_z)*(sim->esp);
            		fprintf(bFile_out, "%f\t %f\t %f\t %lld\n",xf,yf,zf, HostMem->fhd[ix+num_y*(iy+num_z*iz)]);
        	}
        	//fprintf(bFile_out,"\n");
    	}
	}
	fclose (bFile_out);
*/
	return 0;

}


int isnumeric(char a)
{
	if(a>=(char)48 && a<=(char)57) return 1;
	else return 0;
}

int readfloats(int n_floats, float* temp, FILE* pFile)
{
	int ii=0;
	char mystring [200];

	if(n_floats>NFLOATS) return 0; //cannot read more than NFLOATS floats

	while(ii<=0)
	{
		if(feof(pFile)) return 0; //if we reach EOF here something is wrong with the file!
		fgets(mystring , 200 , pFile);
		memset(temp,0,NFLOATS*sizeof(float));
		ii=sscanf(mystring,"%f %f %f %f %f %f %f %f %f %f %f %f",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4],&temp[5],&temp[6],&temp[7],&temp[8],&temp[9],&temp[10],&temp[11]);
		if(ii>n_floats) return 0; //if we read more number than defined something is wrong with the file!
	}
	return 1; // Everyting appears to be ok!
}

int readints(int n_ints, int* temp, FILE* pFile) //replace with template?
{
	int ii=0;
	char mystring[STR_LEN];

	if(n_ints>NINTS) return 0; //cannot read more than NFLOATS floats

	while(ii<=0)
	{
		if(feof(pFile)) return 0; //if we reach EOF here something is wrong with the file!
		fgets(mystring , STR_LEN , pFile);
		memset(temp,0,NINTS*sizeof(int));
		ii=sscanf(mystring,"%d %d %d %d %d",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4]);
		if(ii>n_ints) return 0; //if we read more number than defined something is wrong with the file!
	}
	return 1; // Everyting appears to be ok!
}

int ischar(char a)
{
	if((a>=(char)65 && a<=(char)90)||(a>=(char)97 && a<=(char)122)) return 1;
	else return 0;
}

int read_simulation_data(char* filename, SimulationStruct** simulations, int ignoreAdetection)
{
	int i=0;
	int ii=0;
	unsigned long long number_of_photons;
	unsigned int start_weight;
	int n_simulations = 0;
	int n_layers = 0;
	FILE * pFile;
	char mystring [STR_LEN];
	char str[STR_LEN];
	char AorB;
	float dtot=0;


	float ftemp[NFLOATS];//Find a more elegant way to do this...
	int itemp[NINTS];


	pFile = fopen(filename , "r");
	if (pFile == NULL){perror ("Error opening file");return 0;}

	// First read the first data line (file version) and ignore
	if(!readfloats(1, ftemp, pFile)){perror ("Error reading file version");return 0;}
	//printf("File version: %f\n",ftemp[0]);

	// Second, read the number of runs
	if(!readints(1, itemp, pFile)){perror ("Error reading number of runs");return 0;}
	n_simulations = itemp[0];
	//printf("Number of runs: %d\n",n_simulations);

	// Allocate memory for the SimulationStruct array
	*simulations = (SimulationStruct*) malloc(sizeof(SimulationStruct)*n_simulations);
	if(*simulations == NULL){perror("Failed to malloc simulations.\n");return 0;}//{printf("Failed to malloc simulations.\n");return 0;}

	for(i=0;i<n_simulations;i++)
	{
		// Store the input filename
		strcpy((*simulations)[i].inp_filename,filename);
		// Echo the Filename
		//printf("Input filename: %s\n",filename);

		// Store ignoreAdetection data
		(*simulations)[i].ignoreAdetection=ignoreAdetection;

		// Accumulate fhd
		//(*simulations)[i].fhd_activated = 1;
		// Read the output filename and determine ASCII or Binary output
		ii=0;
		while(ii<=0)
		{
			(*simulations)[i].begin=ftell(pFile);
			fgets (mystring , STR_LEN , pFile);
			ii=sscanf(mystring,"%s %c",str,&AorB);
			if(feof(pFile)|| ii>2){perror("Error reading output filename");return 0;}
			if(ii>0)ii=ischar(str[0]);
		}
		// Echo the Filename and AorB
		//printf("Output filename: %s, AorB=%c\n",str,AorB);
		strcpy((*simulations)[i].outp_filename,str);
		(*simulations)[i].AorB=AorB;

		//printf("begin=%d\n",(*simulations)[i].begin);

		// Read the number of photons
		ii=0;
		while(ii<=0)
		{
			fgets(mystring , STR_LEN , pFile);
			number_of_photons=0;
			ii=sscanf(mystring,"%llu",&number_of_photons);
			if(feof(pFile) || ii>1){perror("Error reading number of photons");return 0;} //if we reach EOF or read more number than defined something is wrong with the file!
		}
		//printf("Number of photons: %lu\n",number_of_photons);
		(*simulations)[i].number_of_photons=(unsigned long long)number_of_photons;
		printf ("\nNumber of excitation photons to be simulated: %llu\n\n", (*simulations)[i].number_of_photons);

		// Read number of photons per voxel for fluorescence simulation
		if(!readints(1, itemp, pFile)){perror ("Error reading number of photons per voxel");return 0;}
		(*simulations)[i].number_of_photons_per_voxel=(unsigned long)itemp[0];

		// Read fluorescence simulation flag
		if(!readints(1, itemp, pFile)){perror ("Error reading fluorescence simulaion flag");return 0;}
		(*simulations)[i].do_fl_sim=itemp[0];
		if (itemp[0]==1 || itemp[0]==2) (*simulations)[i].fhd_activated=1;
			else (*simulations)[i].fhd_activated=0;
		if ((*simulations)[i].do_fl_sim==0) printf("Fluorescence simulation de-activated. \n\n");
			else if ((*simulations)[i].do_fl_sim==1) {
				printf("Fluorescence simulation activated (reflectance). \n");
				printf("Number of photons per voxel to be simulated: %i\n\n", (*simulations)[i].number_of_photons_per_voxel);
			}
			else if ((*simulations)[i].do_fl_sim==2) {
				printf("Fluorescence simulation activated (transmitance). \n");
				printf("Number of photons per voxel to be simulated: %i\n\n", (*simulations)[i].number_of_photons_per_voxel);
			}
			else {perror ("Error reading fluorescence simulaion flag");return 0;}

		// Read dr and dz (3x float)
		if(!readfloats(2, ftemp, pFile)){perror ("Error reading dr and dz");return 0;}
		//printf("dz=%f, dx=%f, dy=%f\n",ftemp[0],ftemp[1],ftemp[2]);
		//(*simulations)[i].det.dz=ftemp[0];
		(*simulations)[i].det.dx=ftemp[0];
		(*simulations)[i].det.dy=ftemp[1];
		//(*simulations)[i].det.dz=ftemp[2];

		// Read No. of dz, dr and da  (3x int)
		if(!readints(2, itemp, pFile)){perror ("Error reading No. of dx, dy and dz");return 0;}

		//printf("No. of dz=%d, dx=%d, dy=%d\n",itemp[0],itemp[1],itemp[2]);
		//(*simulations)[i].det.nz=itemp[0];
		(*simulations)[i].det.nx=itemp[0];
		(*simulations)[i].det.ny=itemp[1];
		//(*simulations)[i].det.nz=itemp[0];

		// Read source position
		if(!readfloats(3, ftemp, pFile)){perror ("Error source position");return 0;}
		printf("Source postion= %f, %f, %f\n\n",ftemp[0],ftemp[1],ftemp[2]);
		//(*simulations)[i].det.nz=itemp[0];
		(*simulations)[i].xi=ftemp[0];
		(*simulations)[i].yi=ftemp[1];
		(*simulations)[i].zi=ftemp[2];
		//(*simulations)[i].det.nz=itemp[0];

		// Leer separacion fuente-detector
		//if(!readfloats(1, ftemp, pFile)){perror ("Error leyendo separacion fuente-detector");return 0;}
		//printf("Useparacion fuente-detector=%f\n",ftemp[0]);
		//(*simulations)[i].det.sep=ftemp[0];

		// Read No. of layers (1xint)
		if(!readints(1, itemp, pFile)){perror ("Error reading No. of layers");return 0;}
		printf("No. of layers=%d\n\n",itemp[0]);
		n_layers = itemp[0];
		(*simulations)[i].n_layers = itemp[0];

		// Allocate memory for the layers (including one for the upper and one for the lower)
		(*simulations)[i].layers = (LayerStruct*) malloc(sizeof(LayerStruct)*(n_layers+2));
		if((*simulations)[i].layers == NULL){perror("Failed to malloc layers.\n");return 0;}//{printf("Failed to malloc simulations.\n");return 0;}

		// Read upper refractive index (1xfloat)
		if(!readfloats(1, ftemp, pFile)){perror ("Error reading upper refractive index");return 0;}
		printf("Upper refractive index=%f\n\n",ftemp[0]);
		(*simulations)[i].layers[0].n=ftemp[0];

		dtot=0;
		for(ii=1;ii<=n_layers;ii++)
		{
			// Read Layer data (5x float)
			if(!readfloats(9, ftemp, pFile)){perror ("Error reading layer data");return 0;}
			//printf("n=%f, mua=%f, mus=%f, flc=%e, AbsCExi=%f, AbsCEmi=%f, eY=%f, g=%f, d=%f\n\n",ftemp[0],ftemp[1],ftemp[2],ftemp[3],ftemp[4],ftemp[5],ftemp[6],ftemp[7],ftemp[8]);
			(*simulations)[i].layers[ii].n=ftemp[0];
			(*simulations)[i].layers[ii].mua=ftemp[1] + ftemp[3] * ftemp[4];
			(*simulations)[i].layers[ii].muaf=ftemp[1] + ftemp[3] * ftemp[5];
			(*simulations)[i].layers[ii].eY=ftemp[6];
			(*simulations)[i].layers[ii].flc=ftemp[3];
			(*simulations)[i].layers[ii].g=ftemp[7];
			(*simulations)[i].layers[ii].z_min=dtot;
			dtot+=ftemp[8];
			(*simulations)[i].layers[ii].z_max=dtot;
			if(ftemp[2]==0.0f)(*simulations)[i].layers[ii].mutr=FLT_MAX; //Glass layer
				else(*simulations)[i].layers[ii].mutr=1.0f/(ftemp[1]+ftemp[2]);
			(*simulations)[i].layers[ii].albedof =  ftemp[2]/(ftemp[3]*ftemp[5] + ftemp[2]);
			printf("Layer %i\n n=%f\n mua=%f\n muaf=%f\n eY=%f\n flc=%e\n g=%f\n z_min=%f\n z_max=%f\n albedof=%f\n AbsCExi=%f\n AbsCEmi=%f\n\n", ii,
						(*simulations)[i].layers[ii].n,
						(*simulations)[i].layers[ii].mua,
						(*simulations)[i].layers[ii].muaf,
						(*simulations)[i].layers[ii].eY,
						(*simulations)[i].layers[ii].flc,
						(*simulations)[i].layers[ii].g,
						(*simulations)[i].layers[ii].z_min,
						(*simulations)[i].layers[ii].z_max,
						(*simulations)[i].layers[ii].albedof,
						ftemp[4],
						ftemp[5]
			);

		}//end ii<n_layers

		//Calcular espesor
		printf("Espesor=%f\n\n",dtot);
		(*simulations)[i].esp=dtot;

		//Direcional photons
		(*simulations)[i].dir=1.0;

		// Read lower refractive index (1xfloat)
		if(!readfloats(1, ftemp, pFile)){perror ("Error reading lower refractive index");return 0;}
		printf("Lower refractive index=%f\n\n",ftemp[0]);
		(*simulations)[i].layers[n_layers+1].n=ftemp[0];

		// Read inclusion data (9xfloat)
		if(!readfloats(12, ftemp, pFile)){perror ("Error leyendo datos de inclusion");return 0;}
		//printf("x=%f, y=%f z=%f, r=%f, n=%f, mua=%f, mus=%f, g=%f\n\n",ftemp[0],ftemp[1],ftemp[2],ftemp[3],ftemp[4],ftemp[5],ftemp[6],ftemp[7]);
		(*simulations)[i].inclusion.x=ftemp[0];
		(*simulations)[i].inclusion.y=ftemp[1];
		(*simulations)[i].inclusion.z=ftemp[2];
		(*simulations)[i].inclusion.r=ftemp[3];
		(*simulations)[i].inclusion.n=ftemp[4];
		(*simulations)[i].inclusion.mua=ftemp[5] + ftemp[7] * ftemp[8];
		(*simulations)[i].inclusion.muaf=ftemp[5] + ftemp[7] * ftemp[9];
		(*simulations)[i].inclusion.flc=ftemp[7];
		(*simulations)[i].inclusion.eY=ftemp[10];
		(*simulations)[i].inclusion.g=ftemp[11];
		if(ftemp[6]==0.0f)(*simulations)[i].inclusion.mutr=FLT_MAX; //Inclusion with mus=0
		else(*simulations)[i].inclusion.mutr=1.0f/(ftemp[5]+ftemp[6]);

		(*simulations)[i].inclusion.albedof =  ftemp[6]/(ftemp[7]*ftemp[9] + ftemp[6]);
		printf("Inclusion\n x=%f \n y=%f\n z=%f\n r=%f\n n=%f\n mua=%f\n muaf=%f\n flc=%e\n eY=%f\n g=%f\n albedof=%f\n AbsCExi=%f\n AbsCEmi=%f\n\n",
					(*simulations)[i].inclusion.x,
					(*simulations)[i].inclusion.y,
					(*simulations)[i].inclusion.z,
					(*simulations)[i].inclusion.r,
					(*simulations)[i].inclusion.n,
					(*simulations)[i].inclusion.mua,
					(*simulations)[i].inclusion.muaf,
					(*simulations)[i].inclusion.flc,
					(*simulations)[i].inclusion.eY,
					(*simulations)[i].inclusion.g,
					(*simulations)[i].inclusion.albedof,
					ftemp[8],
					ftemp[9]
		);

		(*simulations)[i].end=ftell(pFile);

		//calculate start_weight
		double n1=(*simulations)[i].layers[0].n;
		double n2=(*simulations)[i].layers[1].n;
		double r = (n1-n2)/(n1+n2);
		r = r*r;
		printf ("r = %f\n\n",r);
		start_weight = (unsigned int)((double)0xFFFFFFFF*(1-r));
		(*simulations)[i].start_weight=start_weight;

	}//end for i<n_simulations
	return n_simulations;
}
