### Abstract
We present an implementation of a Monte Carlo simulation software for light propagation in fluorescent turbid media, accelerated by GPU (graphic processing unit). The code is based on previous work by Alerstam et al and Wang et al, with the addition of a voxelized medium without symmetries and with an inhomogeneous distribution of absorbers and fluorescent markers.

Broadly speaking, the code developed in the present contribution adds over CUDAMC:

* A complete Cartesian geometry, without assumptions of symmetry, producing x,y Cartesian images as output.
* Inhomogeneities embedded in the medium, defined by an input matrix.
* A voxelization of the bulk media, for inhomogeneity description and for photon hitting density storage.
* The possibility of a isotropic source located anywhere in the medium.
* Fluorescent sources simulation.

### Contact
Nicolás Abel Carbone

Instituto de Física Arroyo Seco – IFAS (UNCPBA) and CIFICEN (UNCPBA-CICPBA-CONICET), Pinto 399, 7000) Tandil, Argentina

ncarbone@exa.unicen.edu.ar