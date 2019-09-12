"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from halotools.utils import crossmatch
from ..source_galaxy_selection import source_galaxy_selection_indices
from ..host_halo_binning import halo_bin_indices


def test1_bijective_case():
    """
    Setup:

    * Each source halo belongs to a unique bin.
    * There exists a unique target halo for every source halo.
    * Each source halo is populated with a single galaxy.

    Verify:

    * The target galaxy catalog is an exact replica of the source galaxy catalog
    """
    num_source_halos = 10
    num_galaxies = num_source_halos
    num_target_halos = num_source_halos
    nhalo_min = 1

    source_halo_dt_list = [(str('halo_id'), str('i8')), (str('bin_number'), str('i4'))]
    source_halos_dtype = np.dtype(source_halo_dt_list)
    source_halos = np.zeros(num_source_halos, dtype=source_halos_dtype)
    source_halos['halo_id'] = np.arange(num_source_halos).astype(int)
    source_halos['bin_number'] = np.arange(num_source_halos).astype(int)

    source_galaxy_dt_list = [(str('host_halo_id'), str('i8'))]
    source_galaxies_dtype = np.dtype(source_galaxy_dt_list)
    source_galaxies = np.zeros(num_galaxies, dtype=source_galaxies_dtype)
    source_galaxies['host_halo_id'] = np.arange(num_galaxies).astype(int)

    target_halo_dt_list = [(str('bin_number'), str('i4')), (str('halo_id'), str('i8'))]
    target_halos_dtype = np.dtype(target_halo_dt_list)
    target_halos = np.zeros(num_target_halos, dtype=target_halos_dtype)
    target_halos['bin_number'] = np.arange(num_target_halos).astype(int)
    target_halos['halo_id'] = np.arange(num_target_halos).astype(int)

    fake_bins = np.arange(-0.5, num_source_halos+0.5, 1)

    _result = source_galaxy_selection_indices(source_galaxies['host_halo_id'],
            source_halos['halo_id'], source_halos['bin_number'], target_halos['bin_number'],
            target_halos['halo_id'], nhalo_min, fake_bins)
    indices, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids = _result

    selected_galaxies = source_galaxies[indices]
    assert len(selected_galaxies) == len(source_galaxies)
    assert np.all(selected_galaxies['host_halo_id'] == source_galaxies['host_halo_id'])
    assert np.all(indices == np.arange(len(indices)))

    assert len(target_galaxy_target_halo_ids) == len(selected_galaxies)
    idxA, idxB = crossmatch(target_galaxy_target_halo_ids, target_halos['halo_id'])
    target_halo_bins = target_halos['bin_number'][idxB]
    assert np.all(np.histogram(target_halo_bins)[0] == np.histogram(source_halos['bin_number'])[0])


def test2_bijective_case():
    """
    Setup:

    * Each source halo belongs to a unique bin.
    * There exists 5 target halos for every source halo.
    * Each source halo is populated with a single galaxy.

    Verify:

    * The target galaxy catalog is a 5x repetition of the source galaxy catalog
    """
    num_source_halos = 10
    num_galaxies = num_source_halos
    num_target_halos = num_source_halos*5
    nhalo_min = 1

    source_halo_dt_list = [(str('halo_id'), str('i8')), (str('bin_number'), str('i4'))]
    source_halos_dtype = np.dtype(source_halo_dt_list)
    source_halos = np.zeros(num_source_halos, dtype=source_halos_dtype)
    source_halos['halo_id'] = np.arange(num_source_halos).astype(int)
    source_halos['bin_number'] = np.arange(num_source_halos).astype(int)

    source_galaxy_dt_list = [(str('host_halo_id'), str('i8'))]
    source_galaxies_dtype = np.dtype(source_galaxy_dt_list)
    source_galaxies = np.zeros(num_galaxies, dtype=source_galaxies_dtype)
    source_galaxies['host_halo_id'] = np.arange(num_galaxies).astype(int)

    target_halo_dt_list = [(str('bin_number'), str('i4')), (str('halo_id'), str('i8'))]
    target_halos_dtype = np.dtype(target_halo_dt_list)
    target_halos = np.zeros(num_target_halos, dtype=target_halos_dtype)
    target_halos['bin_number'] = np.repeat(source_halos['bin_number'], 5)
    target_halos['halo_id'] = np.arange(num_target_halos).astype(int)

    fake_bins = np.arange(-0.5, num_source_halos+0.5, 1)
    # indices, matching_target_halo_ids = source_galaxy_selection_indices(source_galaxies['host_halo_id'],
    #         source_halos['halo_id'], source_halos['bin_number'], target_halos['bin_number'],
    #         target_halos['halo_id'], nhalo_min, fake_bins)
    _result = source_galaxy_selection_indices(source_galaxies['host_halo_id'],
            source_halos['halo_id'], source_halos['bin_number'], target_halos['bin_number'],
            target_halos['halo_id'], nhalo_min, fake_bins)
    indices, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids = _result

    selected_galaxies = source_galaxies[indices]
    assert len(selected_galaxies) == num_target_halos
    assert np.all(selected_galaxies == np.repeat(source_galaxies, 5))

    assert len(target_galaxy_target_halo_ids) == len(selected_galaxies)
    idxA, idxB = crossmatch(target_galaxy_target_halo_ids, target_halos['halo_id'])
    target_halo_bins = target_halos['bin_number'][idxB]
    assert np.all(np.histogram(target_halo_bins)[0] == 5*np.histogram(source_halos['bin_number'])[0])


def test_many_galaxies_per_source_halo():
    """ Test case of mutliple source galaxies per source halo
    """
    #  Set up a source halo catalog with 100 halos in each mass bin
    log_mhost_min, log_mhost_max, dlog_mhost = 10.5, 15.5, 0.5
    log_mhost_bins = np.arange(log_mhost_min, log_mhost_max+dlog_mhost, dlog_mhost)
    log_mhost_mids = 0.5*(log_mhost_bins[:-1] + log_mhost_bins[1:])
    num_distinct_source_halo_masses = len(log_mhost_mids)

    num_source_halos_per_bin = 20
    source_halo_log_mhost = np.tile(log_mhost_mids, num_source_halos_per_bin)
    num_source_halos = len(source_halo_log_mhost)
    source_halo_id = np.arange(num_source_halos).astype(int)
    source_halo_bin_number = halo_bin_indices(log_mhost=(source_halo_log_mhost, log_mhost_bins))
    assert len(source_halo_bin_number) == num_distinct_source_halo_masses*num_source_halos_per_bin, "Bad setup of source_halos"

    ngals_per_source_halo = 3
    num_source_galaxies = num_source_halos*ngals_per_source_halo
    source_galaxy_host_halo_id = np.repeat(source_halo_id, ngals_per_source_halo)
    source_galaxy_host_mass = np.repeat(source_halo_log_mhost, ngals_per_source_halo)
    assert len(source_galaxy_host_mass) == num_source_galaxies, "Bad setup of source_galaxies"

    num_target_halos_per_source_halo = 11
    target_halo_log_mhost = np.repeat(source_halo_log_mhost, num_target_halos_per_source_halo)
    target_halo_bin_number = halo_bin_indices(log_mhost=(target_halo_log_mhost, log_mhost_bins))
    num_target_halos = len(target_halo_bin_number)
    target_halo_ids = np.arange(num_target_halos).astype('i8')

    nhalo_min = 5
    _result = source_galaxy_selection_indices(source_galaxy_host_halo_id,
            source_halo_bin_number, source_halo_id, target_halo_bin_number,
            target_halo_ids, nhalo_min, log_mhost_bins)
    selection_indices, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids = _result

    correct_num_target_galaxies = int(num_target_halos*ngals_per_source_halo)
    assert correct_num_target_galaxies == len(target_galaxy_target_halo_ids) == len(selection_indices)

    idxA, idxB = crossmatch(target_galaxy_target_halo_ids, target_halo_ids)
    assert len(idxA) == len(target_galaxy_target_halo_ids)
    target_halo_bins = target_halo_bin_number[idxB]
    A = num_target_halos_per_source_halo*ngals_per_source_halo
    assert np.all(np.histogram(target_halo_bins)[0] == A*np.histogram(source_halo_bin_number)[0])

    selected_galaxies_target_halo_mass = target_halo_log_mhost[idxB]
    a = halo_bin_indices(log_mhost=(selected_galaxies_target_halo_mass, log_mhost_bins))
    b = halo_bin_indices(log_mhost=(source_galaxy_host_mass, log_mhost_bins))
    assert np.all(np.histogram(a)[0] == num_target_halos_per_source_halo*np.histogram(b)[0])

    selected_galaxies_source_halo_mass = source_galaxy_host_mass[selection_indices]
    assert np.allclose(selected_galaxies_source_halo_mass, selected_galaxies_target_halo_mass)

    gen = zip(selected_galaxies_source_halo_mass, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids)
    for galmass, target_id, source_id in gen:
        source_mask = source_halo_id == source_id
        target_mask = target_halo_ids == target_id
        source_halo_mass = source_halo_log_mhost[source_mask][0]
        target_halo_mass = target_halo_log_mhost[target_mask][0]
        assert source_halo_mass == target_halo_mass == galmass


def test_empty_halos_case():
    """
    """
    #  Set up a source halo catalog with 100 halos in each mass bin
    log_mhost_min, log_mhost_max, dlog_mhost = 10.5, 15.5, 0.5
    log_mhost_bins = np.arange(log_mhost_min, log_mhost_max+dlog_mhost, dlog_mhost)
    log_mhost_mids = 0.5*(log_mhost_bins[:-1] + log_mhost_bins[1:])

    num_source_halos_per_bin = 20
    source_halo_log_mhost = np.tile(log_mhost_mids, num_source_halos_per_bin)
    num_source_halos = len(source_halo_log_mhost)
    source_halo_id = np.arange(num_source_halos).astype(int)
    source_halo_bin_number = halo_bin_indices(log_mhost=(source_halo_log_mhost, log_mhost_bins))

    source_halo_richness = np.tile([0, 3], int(num_source_halos/2))
    source_galaxy_host_halo_id = np.repeat(source_halo_id, source_halo_richness)
    source_galaxy_host_mass = np.repeat(source_halo_log_mhost, source_halo_richness)

    num_target_halos_per_source_halo = 11
    target_halo_bin_number = np.repeat(source_halo_bin_number, num_target_halos_per_source_halo)
    target_halo_log_mhost = np.repeat(source_halo_log_mhost, num_target_halos_per_source_halo)
    num_target_halos = len(target_halo_bin_number)
    target_halo_ids = np.arange(num_target_halos).astype('i8')

    nhalo_min = 5
    _result = source_galaxy_selection_indices(source_galaxy_host_halo_id,
            source_halo_bin_number, source_halo_id, target_halo_bin_number,
            target_halo_ids, nhalo_min, log_mhost_bins)
    selection_indices, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids = _result

    selected_galaxies_source_halo_mass = source_galaxy_host_mass[selection_indices]

    idxA, idxB = crossmatch(target_galaxy_target_halo_ids, target_halo_ids)
    selected_galaxies_target_halo_mass = target_halo_log_mhost[idxB]
    assert np.allclose(selected_galaxies_source_halo_mass, selected_galaxies_target_halo_mass)

    gen = zip(selected_galaxies_source_halo_mass, target_galaxy_target_halo_ids, target_galaxy_source_halo_ids)
    for galmass, target_id, source_id in gen:
        source_mask = source_halo_id == source_id
        target_mask = target_halo_ids == target_id
        source_halo_mass = source_halo_log_mhost[source_mask][0]
        target_halo_mass = target_halo_log_mhost[target_mask][0]
        assert source_halo_mass == target_halo_mass == galmass

