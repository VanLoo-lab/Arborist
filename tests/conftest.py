import simit.simulator as sim
import pytest


@pytest.fixture
def simulate_small():
    gt, simdata = sim.simulate(
        seed=26,
        num_cells=50,
        num_snvs=100,
        coverage=10,
        nclusters=5,
        candidate_trees=None,
        candidate_set_size=4,
        cluster_error_prob=0.05,
        min_proportion=0.05,
    )

    yield simdata


@pytest.fixture
def simulate_large():
    gt, simdata = sim.simulate(
        seed=26,
        num_cells=2000,
        num_snvs=10000,
        coverage=0.1,
        nclusters=5,
        candidate_trees=None,
        candidate_set_size=1,
        cluster_error_prob=0.05,
        min_proportion=0.05,
    )

    yield simdata


@pytest.fixture
def simulate_medium():
    gt, simdata = sim.simulate(
        seed=26,
        num_cells=1000,
        num_snvs=5000,
        coverage=0.05,
        nclusters=10,
        candidate_trees=None,
        candidate_set_size=1,
        cluster_error_prob=0.05,
        min_proportion=0.05,
    )

    yield simdata
