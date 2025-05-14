from arborist.arborist import rank_trees
def test_runs(simulate):
    """
    Test that the simulation runs without errors.
    """
    try:
        likelihoods, best_fit= rank_trees(simulate.candidate_set, simulate.read_counts, alpha=0.001, max_iter=10)
        print(best_fit)
    
    except Exception as e:
        print(f"Arborist failed with error: {e}")
        assert False