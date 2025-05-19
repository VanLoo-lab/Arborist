
import simit.simulator as sim
from arborist.arborist_pandas import rank_trees_pandas
from arborist.arborist import rank_trees



gt, simdata = sim.simulate(
    seed=26,
    num_cells=500,
    num_snvs=1000,
    coverage=1,
    nclusters=5,
    candidate_trees=None,
    candidate_set_size=1,
    cluster_error_prob=0.05,
    min_proportion=0.05
)


likelihood_new, best_fit =rank_trees(simdata.candidate_set, simdata.read_counts, 
                                         alpha=0.001, max_iter=1,verbose=True)

likelihood_old, best_fit =rank_trees_pandas(simdata.candidate_set, simdata.read_counts,
                                          alpha=0.001, max_iter=1,verbose=True)



print("Old likelihoods: ", likelihood_old)
print("New likelihoods: ", likelihood_new)