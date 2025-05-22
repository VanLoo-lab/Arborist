
import simit.simulator as sim
from arborist.arborist_pandas import rank_trees_pandas
from arborist.arborist import rank_trees



gt, simdata = sim.simulate(
    seed=33,
    num_cells=50,
    num_snvs=100,
    coverage=10,
    nclusters=5,
    candidate_trees=None,
    candidate_set_size=10,
    cluster_error_prob=0.0,
    min_proportion=0.02
)


likelihood_new, best_fit =rank_trees(simdata.candidate_set, simdata.read_counts, 
                                         alpha=0.001, max_iter=5,verbose=True)

# likelihood_old, best_fit =rank_trees_pandas(simdata.candidate_set, simdata.read_counts,
#                                           alpha=0.001, max_iter=1,verbose=True)



# print("Old likelihoods: ", likelihood_old)
print("New likelihoods: ", likelihood_new)