import simit.simulator as sim
from arborist.arborist import rank_trees, read_tree_edges_conipher

import pandas as pd 
pth ="/rsrch6/home/genetics/vanloolab/llweber/my_projects/arborist_proj/clonesim_simulations/input/s40_m10000_k5_nseg25_loss0.05/n1000_cov0.1/S10_e0.0"
# gt, simdata = sim.simulate(
#     seed=34,
#     num_cells=500,
#     num_snvs=5000,
#     coverage=0.02,
#     nclusters=10,
#     candidate_trees=None,
#     candidate_set_size=10,
#     cluster_error_prob=0.25,
#     min_proportion=0.02,
#     ad_distance_bound=0.25,
# )

read_counts = pd.read_csv(f"{pth}/read_counts.csv")
snv_clusters = pd.read_csv(f"{pth}/snv_clusters.csv", names=["snv", "cluster"])
print(snv_clusters.head())

read_trees = read_tree_edges_conipher

candidate_trees = read_trees(f"{pth}/candidates.txt", sep=",")

elbos, tfit, all_fits = rank_trees(
    candidate_trees,
    read_counts,
    snv_clusters,
    alpha=0.001,
    verbose=True,
    max_iter=50,
    gamma=1.0,
    update_snvs=True
)

print(elbos)
# print(f"Candidate set size: {len(simdata.candidate_set)}")


# likelihood_new, best_fit, all_tree_fits = rank_trees(
#     simdata.candidate_set,
#     simdata.read_counts,
#     alpha=0.001,
#     max_iter=5,
#     verbose=True,
#     prior=0.7,
#     beta=1,
# )

# # likelihood_old, best_fit =rank_trees_pandas(simdata.candidate_set, simdata.read_counts,
# #                                           alpha=0.001, max_iter=1,verbose=True)


# # print("Old likelihoods: ", likelihood_old)
# print("New likelihoods: ", likelihood_new)
