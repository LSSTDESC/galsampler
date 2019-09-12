# GalSampler
Tools for generating synthetic cosmological data. 

The top-level functions in the source code were called as kernels in constructing the galaxy--halo correspondence used as the basis of the [cosmoDC2](https://arxiv.org/abs/1907.06530) mock galaxy catalog. 

* The `source_halo_index_selection` function sets up a correspondence between host halos in two different N-body simulations. 
* The `source_galaxy_selection_indices` sets up a correspondence between galaxies in the source catalog and host halos in the target catalog. 

### Building the code

To compile the Cython modules:

$ python setup.py build_ext --inplace 
