### Abstract
We present an implementation of a Monte Carlo simulation software for light propagation in fluorescent turbid media, accelerated by GPU (graphic processing unit). The code is based on previous work by Alerstam et al and Wang et al, with the addition of a voxelized medium without symmetries and with an inhomogeneous distribution of absorbers and fluorescent markers.

Broadly speaking, the code developed in the present contribution adds over CUDAMC:

* A complete Cartesian geometry, without assumptions of symmetry, producing x,y Cartesian images as output.
* Inhomogeneities embedded in the medium, defined by an input matrix.
* A voxelization of the bulk media, for inhomogeneity description and for photon hitting density storage.
* The possibility of a isotropic source located anywhere in the medium.
* Fluorescent sources simulation.

### Research paper

This code is discussed in a approved paper published in Biomedical Physics & Engineering Express under the title: "GPU accelerated Monte Carlo simulation of light propagation in inhomogeneous fluorescent turbid media. Application to whole field CW imaging.".

If you are using CUDAMCFL with academic purpose, please cite the preceding paper. In BibTeX form:

```latex
@article{10.1088/2057-1976/aa7b8f,
  author={Nicolás Abel Carbone and Juan A Pomarico and Daniela Inés Iriarte},
    title={GPU accelerated Monte Carlo simulation of light propagation in inhomogeneous fluorescent turbid media. Application to whole field CW imaging.},
  journal={Biomedical Physics & Engineering Express},
        
  url={http://iopscience.iop.org/10.1088/2057-1976/aa7b8f},
  year={2017},
  abstract={Abstract We present an implementation of a Monte Carlo simulation software for fluorescent turbid media, accelerated by GPU (Graphic Processing Unit). &#13; The code is based on previous work by Alerstam et al. and Wang et al., with the addition of a voxelized medium without symmetries and with an inhomogeneous distribution of absorbers and fluorescent markers. &#13; Cartesian coordinates are used in place of the cylindrical ones used in previous versions. &#13; It is particularly aimed at the simulation of CW whole-field reflectance and transmittance images of fluorescence and absorption.&#13; Several tests and comparisons with numerical and theoretical techniques were performed in order to validate our approach.}
```

### Contact
Nicolás Abel Carbone

Instituto de Física Arroyo Seco – IFAS (UNCPBA) and CIFICEN (UNCPBA-CICPBA-CONICET), Pinto 399, 7000) Tandil, Argentina

ncarbone@exa.unicen.edu.ar
