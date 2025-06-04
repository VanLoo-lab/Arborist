import simit.simulator as sim
from arborist.arborist import rank_trees


gt, simdata = sim.simulate(
    seed=34,
    num_cells=500,
    num_snvs=5000,
    coverage=0.02,
    nclusters=10,
    candidate_trees=None,
    candidate_set_size=10,
    cluster_error_prob=0.25,
    min_proportion=0.02,
    ad_distance_bound=0.25,
)

print(f"Candidate set size: {len(simdata.candidate_set)}")


likelihood_new, best_fit, all_tree_fits = rank_trees(
    simdata.candidate_set,
    simdata.read_counts,
    alpha=0.001,
    max_iter=5,
    verbose=True,
    prior=0.7,
    beta=1,
)

# # likelihood_old, best_fit =rank_trees_pandas(simdata.candidate_set, simdata.read_counts,
# #                                           alpha=0.001, max_iter=1,verbose=True)


# # print("Old likelihoods: ", likelihood_old)
# print("New likelihoods: ", likelihood_new)
