"""
"""
import numpy as np

from .source_halo_selection import get_source_bin_from_target_bin


__all__ = ('halo_bin_indices', 'matching_bin_dictionary')


def halo_bin_indices(**haloprop_and_bins_dict):
    """ Calculate a unique cell ID for every host halo.

    Parameters
    ----------
    haloprop_and_bins_dict : dict
        Python dictionary storing the collection of halo properties and bins.
        Each key should be the name of the property to be binned;
        each value should be a two-element tuple storing two ndarrays,
        the first with shape (num_halos, ), the second with shape (nbins, ),
        where ``nbins`` is allowed to vary from property to property.

    Returns
    -------
    cell_ids : ndarray
        Numpy integer array of shape (num_halos, ) storing the integer of the
        (possibly multi-dimensional) bin of each halo.

    Examples
    --------
    In this example, we bin our halos simultaneously by mass and concentration:

    >>> num_halos = 50
    >>> mass = 10**np.random.uniform(10, 15, num_halos)
    >>> conc = np.random.uniform(1, 25, num_halos)
    >>> num_bins_mass, num_bins_conc = 12, 11
    >>> mass_bins = np.logspace(10, 15, num_bins_mass)
    >>> conc_bins = np.logspace(1.5, 20, num_bins_conc)

    >>> cell_ids = halo_bin_indices(mass=(mass, mass_bins), conc=(conc, conc_bins))

    In this case, all values in the ``cell_ids`` array
    will be in the interval [0, num_bins_mass*num_bins_conc).
    """
    bin_indices_dict = {}
    for haloprop_name in haloprop_and_bins_dict.keys():
        arr, bins = haloprop_and_bins_dict[haloprop_name]
        bin_indices = np.maximum(1, np.minimum(np.digitize(arr, bins), len(bins)-1)) - 1
        bin_indices_dict[haloprop_name] = bin_indices

    num_bins_dict = {key: len(haloprop_and_bins_dict[key][1])-1 for key in haloprop_and_bins_dict.keys()}

    return np.ravel_multi_index(list(bin_indices_dict.values()),
            list(num_bins_dict.values()))


def matching_bin_dictionary(assigned_bin_numbers, nmin, bin_shapes):
    """ For every bin number, find the closest bin with more than ``nmin`` objects,
    and return the result in the form of a dictionary.

    Parameters
    ----------
    assigned_bin_numbers : ndarray
        Numpy integer array of shape (num_objects, ) storing the bin number
        to which each object has been assigned

    nmin : int
        Minimum number of objects for the bin to be considered well-sampled

    bin_shapes : tuple
        Sequence storing the dimension of the binning scheme.

        For example, if the bin numbers are calculated using a single
        property binned according to nbins edges, then ``bin_shapes`` = (nbins, )
        If the bin numbers are calculated using two properties with
        nbins1 and nbins2 edges, then ``bin_shapes`` = (nbins1, nbins2)

    Returns
    -------
    matching_bin_dict : dict
        Python dictionary storing the bin correspondence.
        There will be a key for every possible bin number.
        The value bound to that key stores the nearest bin with more than ``nmin`` objects.
    """
    num_bins_total = np.product(bin_shapes)
    unique_bins, counts = np.unique(assigned_bin_numbers, return_counts=True)

    d = {}
    for bin_number in range(num_bins_total):
        if bin_number in unique_bins:
            d[bin_number] = counts[unique_bins == bin_number][0]
        else:
            d[bin_number] = 0

    result = {}
    for i in range(num_bins_total):
        result[i] = get_source_bin_from_target_bin(d, i, nmin, bin_shapes)
    return result
