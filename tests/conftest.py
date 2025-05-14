import simit.simulator as sim
import pytest 


@pytest.fixture 
def simulate():
    gt, simdata = sim.simulate(
        seed=26,
        num_cells=50,
        num_snvs=100,
        coverage=10,
        nclusters=5,
        candidate_trees=None,
        candidate_set_size=4,
        cluster_error_prob=0.05,
        min_proportion=0.05
    )

    yield simdata 