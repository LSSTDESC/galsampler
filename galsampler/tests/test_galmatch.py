"""Unit testing for the galmatch module."""
import numpy as np
from ..galmatch import compute_source_galaxy_selection_indices
from ..galmatch import calculate_indx_correspondence


def test_source_galaxy_selection_indices():
    """Set up a simple example of source and target halo catalogs with a few tightly
    localized collections of properties. Verify that the bookkeeping of the
    compute_source_galaxy_selection_indices function correctly handles the
    correspondence between halos with similar properties.
    """
    n_groupings = 3
    n_source_halos_per_group, n_target_halos_per_group = 50, 600
    n_source_halos = n_groupings * n_source_halos_per_group
    n_target_halos = n_groupings * n_target_halos_per_group

    source_halo_ids = np.arange(n_source_halos).astype(int)
    target_halo_ids = np.arange(n_target_halos).astype(int)

    source_halos_x = np.zeros(n_source_halos)
    target_halos_x = np.zeros(n_target_halos)

    source_halo_richness = np.zeros(n_source_halos).astype(int)
    richness1, richness2, richness3 = 2, 4, 5
    x1, x2, x3 = 1, 10, -10

    i, j = 0, n_source_halos_per_group
    source_halos_x[i:j] = np.random.normal(
        loc=x1, scale=0.01, size=n_source_halos_per_group
    )
    source_halo_richness[i:j] = richness1
    i, j = n_source_halos_per_group, 2 * n_source_halos_per_group
    source_halos_x[i:j] = np.random.normal(
        loc=x2, scale=0.01, size=n_source_halos_per_group
    )
    source_halo_richness[i:j] = richness2
    i, j = 2 * n_source_halos_per_group, 3 * n_source_halos_per_group
    source_halos_x[i:j] = np.random.normal(
        loc=x3, scale=0.01, size=n_source_halos_per_group
    )
    source_halo_richness[i:j] = richness3

    i, j = 0, n_target_halos_per_group
    target_halos_x[i:j] = np.random.normal(
        loc=x1, scale=0.01, size=n_target_halos_per_group
    )
    i, j = n_target_halos_per_group, 2 * n_target_halos_per_group
    target_halos_x[i:j] = np.random.normal(
        loc=x2, scale=0.01, size=n_target_halos_per_group
    )
    i, j = 2 * n_target_halos_per_group, 3 * n_target_halos_per_group
    target_halos_x[i:j] = np.random.normal(
        loc=x3, scale=0.01, size=n_target_halos_per_group
    )

    source_halo_props = (source_halos_x,)
    target_halo_props = (target_halos_x,)

    source_galaxies_host_halo_id = np.repeat(source_halo_ids, source_halo_richness)
    res = compute_source_galaxy_selection_indices(
        source_galaxies_host_halo_id,
        source_halo_ids,
        target_halo_ids,
        source_halo_props,
        target_halo_props,
    )
    selection_indx = res.target_gals_selection_indx
    target_galaxy_target_halo_ids = res.target_gals_target_halo_ids
    n_mock_correct = n_target_halos_per_group * (richness1 + richness2 + richness3)
    assert selection_indx.size == n_mock_correct

    source_gals_host_x = np.repeat(source_halos_x, source_halo_richness)
    target_gals_source_host_x = source_gals_host_x[selection_indx]

    # Check that the mock has inherited from the right halos
    msk1 = target_galaxy_target_halo_ids < n_target_halos_per_group
    dx1 = np.abs(target_gals_source_host_x[msk1] - x1)
    assert np.all(dx1 < 1)

    assert ~np.all(np.abs(target_gals_source_host_x - x1) < 1)

    msk2 = ~msk1 & (target_galaxy_target_halo_ids < 2 * n_target_halos_per_group)
    dx2 = np.abs(target_gals_source_host_x[msk2] - x2)
    assert np.all(dx2 < 1)


def test_calculate_indx_correspondence():
    n_source, n_target_per_source = 10, 5
    n_target = n_source * n_target_per_source
    x_source = np.arange(n_source)

    delta = 0.01
    u_target = np.random.uniform(-delta, delta, n_target)
    x_target = np.repeat(x_source, n_target_per_source) + u_target
    source_props = (x_source,)
    target_props = (x_target,)
    dd_match, indx_match = calculate_indx_correspondence(source_props, target_props)
    assert dd_match.shape == x_target.shape
    assert dd_match.max() <= delta

    x_match = x_source[indx_match]
    assert np.allclose(x_target, x_match, atol=delta)
