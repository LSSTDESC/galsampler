"""
"""
import numpy as np
from halotools.utils import crossmatch, compute_richness

from .source_halo_selection import source_halo_index_selection
from .cython_kernels.galaxy_selection_kernel import galaxy_selection_kernel

__all__ = ('source_galaxy_selection_indices', )


def source_galaxy_selection_indices(source_galaxies_host_halo_id,
            source_halos_bin_number, source_halos_halo_id,
            target_halos_bin_number, target_halo_ids, nhalo_min, *bins):
    """
    Examples
    --------
    source_galaxies_host_halo_id : ndarray
        Numpy integer array of shape (num_source_gals, )
        storing the ID of the host halo of each source galaxy.
        In particular, if the galaxy occupies a subhalo of some larger host halo,
        the value for source_galaxies_host_halo_id of that galaxy should be the
        ID of the larger host halo.

    source_halos_bin_number : ndarray
        Numpy integer array of shape (num_source_halos, )
        storing the bin number assigned to every halo in the source halo catalog.

        Note that it is important to include a *complete* sample of source halos,
        including those that do not host a source galaxy.

        The bin_number can be computed using the `galsampler.halo_bin_indices` function,
        `np.digitize`, or some other means.

    source_halos_halo_id : ndarray
        Numpy integer array of shape (num_source_halos, )
        storing the ID of every halo in the source halo catalog.

        Note that it is important to include a *complete* sample of source halos,
        including those that do not host a source galaxy.

    target_halos_bin_number : ndarray
        Numpy integer array of shape (num_target_halos, )
        storing the bin number assigned to every halo in the target halo catalog.

        The bin_number can be computed using the `galsampler.halo_bin_indices` function,
        `np.digitize`, or some other means.

    target_halo_ids : ndarray
        Numpy integer array of shape (num_target_halos, )
        storing the ID of every halo in the target halo catalog.

    nhalo_min : int
        Minimum permissible number of halos in source catalog for a cell to be
        considered well-sampled

    *bins : sequence
        Sequence of arrays that were used to bin the halos

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_target_gals, ) storing the indices
        of the selected galaxies

    target_galaxy_target_halo_ids : ndarray
        Numpy integer array of shape (num_target_gals, ) storing the halo ID
        of the target halo hosting each selected source galaxy

    target_galaxy_source_halo_ids : ndarray
        Numpy integer array of shape (num_target_gals, ) storing the halo ID
        of the source halo hosting each selected source galaxy
    """
    #  Sort the source galaxies so that members of a common halo are grouped together
    idx_sorted_source_galaxies = np.argsort(source_galaxies_host_halo_id)
    sorted_source_galaxies_host_halo_id = source_galaxies_host_halo_id[idx_sorted_source_galaxies]

    #  Calculate the index correspondence array that will undo the sorting at the end
    num_source_gals = len(source_galaxies_host_halo_id)
    idx_unsorted_galaxy_indices = np.arange(num_source_gals).astype('i8')[idx_sorted_source_galaxies]

    #  For each source halo, calculate the number of resident galaxies
    source_halos_richness = compute_richness(
                source_halos_halo_id, sorted_source_galaxies_host_halo_id)

    #  For each source halo, calculate the index of its first resident galaxy
    source_halo_sorted_source_galaxies_indices = _galaxy_table_indices(
                source_halos_halo_id, sorted_source_galaxies_host_halo_id)

    #  For each target halo, calculate the index of the associated source halo
    source_halo_selection_indices, matching_target_halo_ids = source_halo_index_selection(
            source_halos_bin_number, target_halos_bin_number, target_halo_ids, nhalo_min, *bins)

    #  For each target halo, calculate the number of galaxies
    target_halo_richness = source_halos_richness[source_halo_selection_indices]
    num_target_gals = np.sum(target_halo_richness)

    #  For each target halo, calculate the halo ID of the associated source halo
    target_halo_source_halo_ids = source_halos_halo_id[source_halo_selection_indices]

    #  For each target halo, calculate the index of its first resident galaxy
    target_halo_first_sorted_source_gal_indices = (
                source_halo_sorted_source_galaxies_indices[source_halo_selection_indices])

    #  For every target halo, we know the index of the first and last galaxy to select
    #  Calculate an array of shape (num_target_gals, ) with the index of each selected galaxy
    sorted_source_galaxy_selection_indices = np.array(galaxy_selection_kernel(
            target_halo_first_sorted_source_gal_indices.astype('i8'),
            target_halo_richness.astype('i4'), num_target_gals))

    #  For each target galaxy, calculate the halo ID of its source and target halo
    target_galaxy_target_halo_ids = np.repeat(matching_target_halo_ids, target_halo_richness)
    target_galaxy_source_halo_ids = np.repeat(target_halo_source_halo_ids, target_halo_richness)

    #  For each index in the sorted galaxy catalog,
    #  calculate the index of the catalog in its original order
    selection_indices = idx_unsorted_galaxy_indices[sorted_source_galaxy_selection_indices]

    return (selection_indices, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids)


def _galaxy_table_indices(source_halo_id, galaxy_host_halo_id):
    """ For every halo in the source halo catalog, calculate the index
    in the source galaxy catalog of the first appearance of a galaxy that
    occupies the halo, reserving -1 for source halos with no resident galaxies.

    Parameters
    ----------
    source_halo_id : ndarray
        Numpy integer array of shape (num_halos, )

    galaxy_host_halo_id : ndarray
        Numpy integer array of shape (num_gals, )

    Returns
    -------
    indices : ndarray
        Numpy integer array of shape (num_halos, ).
        All values will be in the interval [-1, num_gals)
    """
    uval_gals, indx_uval_gals = np.unique(galaxy_host_halo_id, return_index=True)
    idxA, idxB = crossmatch(source_halo_id, uval_gals)
    num_source_halos = len(source_halo_id)
    indices = np.zeros(num_source_halos) - 1
    indices[idxA] = indx_uval_gals[idxB]
    return indices.astype(int)

